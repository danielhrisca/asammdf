------------
Bus logging
------------

Initial read only mode for mdf version 4.10 files containing CAN/LIN bus logging
is now implemented.

To handle this, the **canmatrix** package was added to the dependencies; you will need to install the latest code
from the **canmatrix** library.

Let's take for example the following situation: the .dbc contains the definition
for the CAN message called "VehicleStatus" with `ID=123`. This message contains the
signal "EngineStatus". Logging was made from the CAN bus with `ID=1` (CAN1).

There multiple ways to address this channel in this situation:

#. short signal name as found in the .dbc file 

    .. code:: python
    
        mdf.get('EngineStatus')
        
#. dbc message name and short signal name, delimited by dot

    .. code:: python
    
        mdf.get('VehicleStatus.EngineStatus')     
        
#. CAN bus ID, dbc message name and short signal name, delimited by dot

    .. code:: python
    
        mdf.get('CAN1.VehicleStatus.EngineStatus')    
        
#. ASAM conformant message ID and short signal name, delimited by dot

    .. code:: python
    
        mdf.get('CAN_DataFrame_123.EngineStatus')     
        
#. CAN bus ID, ASAM conformant message ID and short signal name, delimited by dot

    .. code:: python

        mdf.get('CAN1.CAN_DataFrame_123.EngineStatus')


Container I-PDUs
================

AUTOSAR **container I-PDUs** pack several smaller PDUs into a single CAN(-FD)
frame. When the database describes a frame as a PDU container, each contained
PDU is decoded and gets **its own channel group**, so its signals are addressed
exactly like any other bus logging signal:

.. code:: python

    mdf.get('EngineStatus')                       # short signal name
    mdf.get('CAN1.VehicleStatus.EngineStatus')    # container frame name

Both container layouts are supported: *dynamic* containers, where every
contained PDU is prefixed by a header (``Header_ID`` + ``Header_DLC``) and its
offset therefore changes from frame to frame, and *static* containers, where the
contained PDUs sit at fixed offsets.

A sender may transmit a contained PDU shorter than the length declared in the
database — the header DLC is the authority. Signals that reach past the
transmitted length would otherwise be decoded out of the neighbouring PDU or out
of the container padding, so those samples are marked invalid:

.. code:: python

    sig = mdf.get('SomeSignal')
    sig.invalidation_bits    # None, or True where the bytes were not transmitted

Two limitations are worth noting: a multiplexed *contained* PDU is not modelled
by **canmatrix**, so it cannot be decoded, and container I-PDUs are only handled
for CAN — not for LIN.


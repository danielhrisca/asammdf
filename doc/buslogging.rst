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
        

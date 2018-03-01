# Python version
_Please run the following snippet and write the output here_
```python
import platform
import sys
from pprint import pprint

pprint("python=" + sys.version)
pprint("os=" + platform.platform())

try:
    import numpy
    pprint("numpy=" + numpy.__version__)
except ImportError:
    pass

try:
    import asammdf
    pprint("asammdf=" + asammdf.__version__)
except ImportError:
    pass
```
# Code 

  ## Code snippet
  _please write here the code snippet that triggers the error_
  
  ## Traceback
  _pleaase write here the error traceback_
  
# Description
_Please describe the issue here._

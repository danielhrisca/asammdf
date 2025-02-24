# Python version

_Please run the following snippet and write the output here_

```python
import platform
from pprint import pprint
import sys

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

## MDF version

_please write here the file version (you can run `print(MDF(file).version)`)_

## Code snippet

_please write here the code snippet that triggers the error_

## Traceback

_please write here the error traceback_

# Description

The fastest way to debug is to have the original file. For data protection you can use the static
method _scramble_ to scramble all text blocks, and send the scrambled file by e-mail.

_Please describe the issue here._

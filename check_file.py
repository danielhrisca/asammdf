import asammdf
from pathlib import Path
import matplotlib.pyplot as plt

print("Using asammdf==%s" % asammdf.__version__)

f = Path("test.mf4") 
mdf_handle = asammdf.MDF4(f)

signal = mdf_handle.get("C1_VehicleSpeedVSOSig")

t, v = signal.timestamps, signal.samples

plt.figure(figsize=(15, 5))
plt.plot(t, v)
plt.savefig("test_%s.png" % asammdf.__version__, dpi=200)

print("Done!")

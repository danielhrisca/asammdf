"""
*asammdf* MDF usage example
"""

import numpy as np

from asammdf import MDF, Signal

# create 3 Signal objects

timestamps = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)

# unit8
s_uint8 = Signal(
    samples=np.array([0, 1, 2, 3, 4], dtype=np.uint8),
    timestamps=timestamps,
    name="Uint8_Signal",
    unit="u1",
)
# int32
s_int32 = Signal(
    samples=np.array([-20, -10, 0, 10, 20], dtype=np.int32),
    timestamps=timestamps,
    name="Int32_Signal",
    unit="i4",
)

# float64
s_float64 = Signal(
    samples=np.array([-20, -10, 0, 10, 20], dtype=np.float64),
    timestamps=timestamps,
    name="Float64_Signal",
    unit="f8",
)

# create empty MDf version 4.00 file
with MDF(version="4.10") as mdf4:
    # append the 3 signals to the new file
    signals = [s_uint8, s_int32, s_float64]
    mdf4.append(signals, comment="Created by Python")

    # save new file
    mdf4.save("my_new_file.mf4", overwrite=True)

    # convert new file to mdf version 3.10
    mdf3 = mdf4.convert(version="3.10")
    print(mdf3.version)

    # get the float signal
    sig = mdf3.get("Float64_Signal")
    print(sig)

    # cut measurement from 0.3s to end of measurement
    mdf4_cut = mdf4.cut(start=0.3)
    mdf4_cut.get("Float64_Signal").plot()

    # cut measurement from start of measurement to 0.4s
    mdf4_cut = mdf4.cut(stop=0.45)
    mdf4_cut.get("Float64_Signal").plot()

    # filter some signals from the file
    mdf4 = mdf4.filter(["Int32_Signal", "Uint8_Signal"])

    # save using zipped transpose deflate blocks
    mdf4.save("out.mf4", compression=2, overwrite=True)

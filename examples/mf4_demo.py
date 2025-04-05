from typing import Any

import numpy as np
import numpy.typing as npt

from asammdf import MDF, Signal

cycles = 100
sigs = []

mdf = MDF()

t = np.arange(cycles, dtype=np.float64)

# no conversion
sig = Signal(
    np.ones(cycles, dtype=np.uint64),
    t,
    name="Channel_no_conversion",
    unit="s",
    conversion=None,
    comment="Unsigned 64 bit channel {}",
)
sigs.append(sig)

# linear
conversion: dict[str, object] = {"a": 2, "b": -0.5}
sig = Signal(
    np.ones(cycles, dtype=np.int64),
    t,
    name="Channel_linear_conversion",
    unit="Nm",
    conversion=conversion,
    comment="Signed 64bit channel with linear conversion",
)
sigs.append(sig)


# algebraic
conversion = {"formula": "2 * sin(X)"}
sig = Signal(
    np.arange(cycles, dtype=np.int32) / 100.0,
    t,
    name="Channel_algebraic",
    unit="eV",
    conversion=conversion,
    comment="Sinus channel with algebraic conversion",
)
sigs.append(sig)

# rational
conversion = {"P1": 0, "P2": 4, "P3": -0.5, "P4": 0, "P5": 0, "P6": 1}
sig = Signal(
    np.ones(cycles, dtype=np.int64),
    t,
    name="Channel_rational_conversion",
    unit="Nm",
    conversion=conversion,
    comment="Channel with rational conversion",
)
sigs.append(sig)

# string channel
strings = [f"String channel sample {j}".encode("ascii") for j in range(cycles)]
sig = Signal(
    np.array(strings),
    t,
    name="Channel_string",
    comment="String channel",
    encoding="latin-1",
)
sigs.append(sig)

# byte array
ones = np.ones(cycles, dtype=np.dtype("(8,)u1"))
sig = Signal(ones * 111, t, name="Channel_bytearay", comment="Byte array channel")
sigs.append(sig)

# tabular
vals = 20
conversion = {f"raw_{i}": i for i in range(vals)}
conversion.update({f"phys_{i}": -i for i in range(vals)})
sig = Signal(
    np.arange(cycles, dtype=np.uint32) % 20,
    t,
    name="Channel_tabular",
    unit="-",
    conversion=conversion,
    comment="Tabular channel",
)
sigs.append(sig)

# value to text
vals = 20
conversion = {f"val_{i}": i for i in range(vals)}
conversion.update({f"text_{i}": f"key_{i}".encode("ascii") for i in range(vals)})
conversion["default"] = b"default key"
sig = Signal(
    np.arange(cycles, dtype=np.uint32) % 30,
    t,
    name="Channel_value_to_text",
    conversion=conversion,
    comment="Value to text channel",
)
sigs.append(sig)

# tabular with range
vals = 20
conversion = {f"lower_{i}": i * 10 for i in range(vals)}
conversion.update({f"upper_{i}": (i + 1) * 10 for i in range(vals)})
conversion.update({f"phys_{i}": i for i in range(vals)})
conversion["default"] = -1
sig = Signal(
    2 * np.arange(cycles, dtype=np.float64),
    t,
    name="Channel_value_range_to_value",
    unit="order",
    conversion=conversion,
    comment="Value range to value channel",
)
sigs.append(sig)

# value range to text
vals = 20
conversion = {f"lower_{i}": i * 10 for i in range(vals)}
conversion.update({f"upper_{i}": (i + 1) * 10 - 5 for i in range(vals)})
conversion.update({f"text_{i}": f"Level {i}" for i in range(vals)})
conversion["default"] = b"Unknown level"
sig = Signal(
    6 * np.arange(cycles, dtype=np.uint64) % 240,
    t,
    name="Channel_value_range_to_text",
    conversion=conversion,
    comment="Value range to text channel",
)
sigs.append(sig)


mdf.append(sigs, comment="single dimensional channels", common_timebase=True)


sigs = []

# lookup tabel with axis
samples: list[npt.NDArray[Any]] = [
    np.ones((cycles, 2, 3), dtype=np.uint64) * 1,
    np.ones((cycles, 2), dtype=np.uint64) * 2,
    np.ones((cycles, 3), dtype=np.uint64) * 3,
]

types: list[npt.DTypeLike] = [
    ("Channel_lookup_with_axis", "(2, 3)<u8"),
    ("channel_axis_1", "(2, )<u8"),
    ("channel_axis_2", "(3, )<u8"),
]

sig = Signal(
    np.rec.fromarrays(samples, dtype=np.dtype(types)),
    t,
    name="Channel_lookup_with_axis",
    unit="A",
    comment="Array channel with axis",
)
sigs.append(sig)

# lookup tabel with default axis
samples = [np.ones((cycles, 2, 3), dtype=np.uint64) * 4]

types = [("Channel_lookup_with_default_axis", "(2, 3)<u8")]

sig = Signal(
    np.rec.fromarrays(samples, dtype=np.dtype(types)),
    t,
    name="Channel_lookup_with_default_axis",
    unit="mA",
    comment="Array channel with default axis",
)
sigs.append(sig)

# structure channel composition
samples = [
    np.ones(cycles, dtype=np.uint8) * 10,
    np.ones(cycles, dtype=np.uint16) * 20,
    np.ones(cycles, dtype=np.uint32) * 30,
    np.ones(cycles, dtype=np.uint64) * 40,
    np.ones(cycles, dtype=np.int8) * -10,
    np.ones(cycles, dtype=np.int16) * -20,
    np.ones(cycles, dtype=np.int32) * -30,
    np.ones(cycles, dtype=np.int64) * -40,
]

types = [
    ("struct_channel_0", np.uint8),
    ("struct_channel_1", np.uint16),
    ("struct_channel_2", np.uint32),
    ("struct_channel_3", np.uint64),
    ("struct_channel_4", np.int8),
    ("struct_channel_5", np.int16),
    ("struct_channel_6", np.int32),
    ("struct_channel_7", np.int64),
]

sig = Signal(
    np.rec.fromarrays(samples, dtype=np.dtype(types)),
    t,
    name="Channel_structure_composition",
    comment="Structure channel composition",
)
sigs.append(sig)


# nested structures
arrays = [
    np.ones(cycles, dtype=np.float64) * 41,
    np.ones(cycles, dtype=np.float64) * 42,
    np.ones(cycles, dtype=np.float64) * 43,
    np.ones(cycles, dtype=np.float64) * 44,
]

types = [
    ("level41", np.float64),
    ("level42", np.float64),
    ("level43", np.float64),
    ("level44", np.float64),
]

l4_arr = np.rec.fromarrays(arrays, dtype=types)

arrays = [
    l4_arr,
    l4_arr,
    l4_arr,
]

types = [
    ("level31", l4_arr.dtype),
    ("level32", l4_arr.dtype),
    ("level33", l4_arr.dtype),
]

l3_arr = np.rec.fromarrays(arrays, dtype=types)


arrays = [
    l3_arr,
    l3_arr,
]

types = [("level21", l3_arr.dtype), ("level22", l3_arr.dtype)]

l2_arr = np.rec.fromarrays(arrays, dtype=types)


arrays = [l2_arr]

types = [("level11", l2_arr.dtype)]

l1_arr = np.rec.fromarrays(arrays, dtype=types)


sigs.append(Signal(l1_arr, t, name="Nested_structures"))

mdf.append(sigs, comment="arrays", common_timebase=True)

mdf.save("demo.mf4", overwrite=True)

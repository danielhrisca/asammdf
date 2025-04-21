from pathlib import Path

import numpy as np
import numpy.typing as npt
from typing_extensions import Any

from asammdf import MDF, Signal, SUPPORTED_VERSIONS
import asammdf.blocks.v2_v3_blocks as v3b
import asammdf.blocks.v2_v3_constants as v3c
import asammdf.blocks.v4_blocks as v4b
import asammdf.blocks.v4_constants as v4c

SUPPORTED_VERSIONS = SUPPORTED_VERSIONS[1:]  # type: ignore[misc]

cycles = 500
channels_count = 20
array_channels_count = 20


def get_test_data(filename: str = "") -> Path:
    """
    Utility functions needed by all test scripts.
    """
    return Path(__file__).resolve().parent.joinpath("/data/", filename)


def generate_test_file(tmpdir: str, version: str = "4.10") -> Path | None:
    mdf = MDF(version=version)

    if version <= "3.30":
        filename = Path(tmpdir) / f"big_test_{version}.mdf"
    else:
        filename = Path(tmpdir) / f"big_test_{version}.mf4"

    if filename.exists():
        return filename

    t = np.arange(cycles, dtype=np.float64)

    cls = v4b.ChannelConversion if version >= "4.00" else v3b.ChannelConversion

    # no conversion
    sigs = []
    for i in range(channels_count):
        sig = Signal(
            np.ones(cycles, dtype=np.uint64) * i,
            t,
            name=f"Channel_{i}",
            unit=f"unit_{i}",
            conversion=None,
            comment=f"Unsigned int 16bit channel {i}",
            raw=True,
        )
        sigs.append(sig)
    mdf.append(sigs, common_timebase=True)

    # linear
    sigs = []
    for i in range(channels_count):
        conversion: dict[str, Any] = {
            "conversion_type": v4c.CONVERSION_TYPE_LIN if version >= "4.00" else v3c.CONVERSION_TYPE_LINEAR,
            "a": float(i),
            "b": -0.5,
        }
        sig = Signal(
            np.ones(cycles, dtype=np.int64),
            t,
            name=f"Channel_{i}",
            unit=f"unit_{i}",
            conversion=cls(**conversion),
            comment=f"Signed 16bit channel {i} with linear conversion",
            raw=True,
        )
        sigs.append(sig)
    mdf.append(sigs, common_timebase=True)

    # algebraic
    sigs = []
    for i in range(channels_count):
        conversion = {
            "conversion_type": v4c.CONVERSION_TYPE_ALG if version >= "4.00" else v3c.CONVERSION_TYPE_FORMULA,
            "formula": f"{i} * sin(X)",
        }
        sig = Signal(
            np.arange(cycles, dtype=np.int32) / 100.0,
            t,
            name=f"Channel_{i}",
            unit=f"unit_{i}",
            conversion=cls(**conversion),
            comment=f"Sinus channel {i} with algebraic conversion",
            raw=True,
        )
        sigs.append(sig)
    mdf.append(sigs, common_timebase=True)

    # rational
    sigs = []
    for i in range(channels_count):
        conversion = {
            "conversion_type": v4c.CONVERSION_TYPE_RAT if version >= "4.00" else v3c.CONVERSION_TYPE_RAT,
            "P1": 0,
            "P2": i,
            "P3": -0.5,
            "P4": 0,
            "P5": 0,
            "P6": 1,
        }
        sig = Signal(
            np.ones(cycles, dtype=np.int64),
            t,
            name=f"Channel_{i}",
            unit=f"unit_{i}",
            conversion=cls(**conversion),
            comment=f"Channel {i} with rational conversion",
            raw=True,
        )
        sigs.append(sig)
    mdf.append(sigs, common_timebase=True)

    # string
    sigs = []
    encoding = "latin-1" if version < "4.00" else "utf-8"
    for i in range(channels_count):
        strings = [f"Channel {i} sample {j}".encode(encoding) for j in range(cycles)]
        sig = Signal(
            np.array(strings),
            t,
            name=f"Channel_{i}",
            unit=f"unit_{i}",
            comment=f"String channel {i}",
            raw=True,
            encoding=encoding,
        )
        sigs.append(sig)
    mdf.append(sigs, common_timebase=True)

    # byte array
    sigs = []
    ones = np.ones(cycles, dtype=np.dtype("(8,)u1"))
    for i in range(channels_count):
        sig = Signal(
            ones * i,
            t,
            name=f"Channel_{i}",
            unit=f"unit_{i}",
            comment=f"Byte array channel {i}",
            raw=True,
        )
        sigs.append(sig)
    mdf.append(sigs, common_timebase=True)

    # value to text
    sigs = []
    ones = np.ones(cycles, dtype=np.uint64)
    conversion = {
        "raw": np.arange(255, dtype=np.float64),
        "phys": np.array([f"Value {i}".encode("ascii") for i in range(255)]),
        "conversion_type": v4c.CONVERSION_TYPE_TABX if version >= "4.00" else v3c.CONVERSION_TYPE_TABX,
        "links_nr": 260,
        "ref_param_nr": 255,
    }

    for i in range(255):
        conversion[f"val_{i}"] = conversion[f"param_val_{i}"] = conversion["raw"][i]
        conversion[f"text_{i}"] = conversion["phys"][i]
    conversion[f"text_{255}"] = "Default"

    for i in range(channels_count):
        sig = Signal(
            ones * i,
            t,
            name=f"Channel_{i}",
            unit=f"unit_{i}",
            comment=f"Value to text channel {i}",
            conversion=cls(**conversion),
            raw=True,
        )
        sigs.append(sig)
    mdf.append(sigs, common_timebase=True)

    name = mdf.save(filename, overwrite=True)
    mdf.close()

    return None


def generate_arrays_test_file(tmpdir: str) -> Path | None:
    version = "4.10"
    mdf = MDF(version=version)
    filename = Path(tmpdir) / f"arrays_test_{version}.mf4"

    if filename.exists():
        return filename

    t = np.arange(cycles, dtype=np.float64)

    # lookup tabel with axis
    sigs = []
    for i in range(array_channels_count):
        samples: list[npt.NDArray[Any]] = [
            np.ones((cycles, 2, 3), dtype=np.uint64) * i,
            np.ones((cycles, 2), dtype=np.uint64) * i,
            np.ones((cycles, 3), dtype=np.uint64) * i,
        ]

        types: list[npt.DTypeLike] = [
            (f"Channel_{i}", "(2, 3)<u8"),
            (f"channel_{i}_axis_1", "(2, )<u8"),
            (f"channel_{i}_axis_2", "(3, )<u8"),
        ]

        sig = Signal(
            np.rec.fromarrays(samples, dtype=np.dtype(types)),
            t,
            name=f"Channel_{i}",
            unit=f"unit_{i}",
            conversion=None,
            comment=f"Array channel {i}",
            raw=True,
        )
        sigs.append(sig)
    mdf.append(sigs, common_timebase=True)

    # lookup tabel with default axis
    sigs = []
    for i in range(array_channels_count):
        samples = [np.ones((cycles, 2, 3), dtype=np.uint64) * i]

        types = [(f"Channel_{i}", "(2, 3)<u8")]

        sig = Signal(
            np.rec.fromarrays(samples, dtype=np.dtype(types)),
            t,
            name=f"Channel_{i}",
            unit=f"unit_{i}",
            conversion=None,
            comment=f"Array channel {i} with default axis",
            raw=True,
        )
        sigs.append(sig)
    mdf.append(sigs, common_timebase=True)

    # structure channel composition
    sigs = []
    for i in range(array_channels_count):
        samples = [
            np.ones(cycles, dtype=np.uint8) * i,
            np.ones(cycles, dtype=np.uint16) * i,
            np.ones(cycles, dtype=np.uint32) * i,
            np.ones(cycles, dtype=np.uint64) * i,
            np.ones(cycles, dtype=np.int8) * i,
            np.ones(cycles, dtype=np.int16) * i,
            np.ones(cycles, dtype=np.int32) * i,
            np.ones(cycles, dtype=np.int64) * i,
        ]

        types = [
            (f"struct_{i}_channel_0", np.uint8),
            (f"struct_{i}_channel_1", np.uint16),
            (f"struct_{i}_channel_2", np.uint32),
            (f"struct_{i}_channel_3", np.uint64),
            (f"struct_{i}_channel_4", np.int8),
            (f"struct_{i}_channel_5", np.int16),
            (f"struct_{i}_channel_6", np.int32),
            (f"struct_{i}_channel_7", np.int64),
        ]

        sig = Signal(
            np.rec.fromarrays(samples, dtype=np.dtype(types)),
            t,
            name=f"Channel_{i}",
            unit=f"unit_{i}",
            conversion=None,
            comment=f"Structure channel composition {i}",
            raw=True,
        )
        sigs.append(sig)

    mdf.append(sigs, common_timebase=True)

    name = mdf.save(filename, overwrite=True)

    mdf.close()

    return None


if __name__ == "__main__":
    #    generate_test_file("3.30")
    #    generate_test_file("4.10")
    generate_arrays_test_file(r"D:\TMP")

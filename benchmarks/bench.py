"""
benchmark asammdf vs mdfreader
"""

import argparse
from collections.abc import Callable, Iterable
from io import StringIO
import multiprocessing
from multiprocessing.connection import Connection
import os
import platform
import sys
import traceback
from types import TracebackType
import typing
from typing import Literal

from mdfreader import __version__ as mdfreader_version
from mdfreader import Mdf as MDFreader
import numpy as np
import psutil
from typing_extensions import Any

from asammdf import __version__ as asammdf_version
from asammdf import MDF, Signal
from asammdf.blocks import mdf_v3, mdf_v4
import asammdf.blocks.v2_v3_blocks as v3b
import asammdf.blocks.v2_v3_constants as v3c
import asammdf.blocks.v4_blocks as v4b
import asammdf.blocks.v4_constants as v4c

try:
    import resource
except ImportError:
    pass

PYVERSION = sys.version_info[0]

from time import perf_counter


class MyList(list[str]):
    """list that prints the items that are appended or extended"""

    def append(self, item: str) -> None:
        """append item and print it to stdout"""
        print(item)
        super().append(item)

    def extend(self, items: Iterable[str]) -> None:
        """extend items and print them to stdout
        using the new line separator
        """
        print("\n".join(items))
        super().extend(items)


class Timer:
    """measures the RAM usage and elased time. The information is saved in
    the output attribute and any Exception text is saved in the error attribute

    Parameters
    ----------
    topic : str
        timer title; only used if Exceptions are raised during execution
    message : str
        execution item description
    fmt : str
        output fmt; can be "rst" (rstructured text) or "md" (markdown)
    """

    def __init__(self, topic: str, message: str, fmt: Literal["rst", "md"] = "rst") -> None:
        self.topic = topic
        self.message = message
        self.output = ""
        self.error = ""
        self.fmt = fmt
        self.start = 0.0

    def __enter__(self) -> "Timer":
        self.start = perf_counter()
        return self

    def __exit__(
        self, type_: type[BaseException] | None, value: BaseException | None, tracebackobj: TracebackType | None
    ) -> bool | None:
        elapsed_time = int((perf_counter() - self.start) * 1000)
        process = psutil.Process(os.getpid())

        if platform.system() == "Windows":
            peak_wset: int = getattr(process.memory_info(), "peak_wset")  # noqa: B009
            ram_usage = int(peak_wset / 1024 / 1024)
        else:
            ram_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # type: ignore[attr-defined, unused-ignore]
            ram_usage = int(ram_usage / 1024)

        if tracebackobj:
            info_io = StringIO()
            traceback.print_tb(tracebackobj, None, info_io)
            info_io.seek(0)
            info = info_io.read()
            self.error = f"{self.topic} : {self.message}\n{type_}\t \n{value}{info}"
            if self.fmt == "rst":
                self.output = "{:<50} {:>9} {:>8}".format(self.message, "0*", "0*")
            elif self.fmt == "md":
                self.output = "|{:<50}|{:>9}|{:>8}|".format(self.message, "0*", "0*")
        else:
            if self.fmt == "rst":
                self.output = f"{self.message:<50} {elapsed_time:>9} {ram_usage:>8}"
            elif self.fmt == "md":
                self.output = f"|{self.message:<50}|{elapsed_time:>9}|{ram_usage:>8}|"

        return True


def generate_test_files(version: str = "4.10") -> str | None:
    cycles = 3000
    channels_count = 2000
    mdf = MDF(version=version)

    if version <= "3.30":
        filename = r"test.mdf"
    else:
        filename = r"test.mf4"

    if os.path.exists(filename):
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
        conversion: Any = {
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
    for i in range(channels_count):
        strings = [f"Channel {i} sample {j}".encode("ascii") for j in range(cycles)]
        sig = Signal(
            np.array(strings),
            t,
            name=f"Channel_{i}",
            unit=f"unit_{i}",
            comment=f"String channel {i}",
            raw=True,
            encoding="utf-8",
        )
        sigs.append(sig)
    mdf.append(sigs, common_timebase=True)

    # byte array
    sigs = []
    ones = np.ones(cycles, dtype=np.dtype("(8,)u1"))
    for i in range(channels_count):
        sig = Signal(
            ones * (i % 255),
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

    mdf.save(filename, overwrite=True)

    return None


def open_mdf3(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    with Timer("Open file", f"asammdf {asammdf_version} mdfv3", fmt) as timer:
        MDF(r"test.mdf")
    output.send([timer.output, timer.error])


def open_mdf4(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    with Timer("Open file", f"asammdf {asammdf_version} mdfv4", fmt) as timer:
        MDF(r"test.mf4")
    output.send([timer.output, timer.error])


def open_mdf4_column(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    with Timer("Open file", f"asammdf {asammdf_version} column mdfv4", fmt) as timer:
        MDF(r"test_column.mf4")
    output.send([timer.output, timer.error])


def save_mdf3(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    x = MDF(r"test.mdf")
    with Timer("Save file", f"asammdf {asammdf_version} mdfv3", fmt) as timer:
        x.save(r"x.mdf", overwrite=True)
    output.send([timer.output, timer.error])


def save_mdf4(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    x = MDF(r"test.mf4")
    with Timer("Save file", f"asammdf {asammdf_version} mdfv4", fmt) as timer:
        x.save(r"x.mf4", overwrite=True)
    output.send([timer.output, timer.error])


def save_mdf4_column(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    x = MDF(r"test_column.mf4")
    with Timer("Save file", f"asammdf {asammdf_version} mdfv4 column", fmt) as timer:
        x.save(r"x.mf4", overwrite=True)
    output.send([timer.output, timer.error])


def get_all_mdf3(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    x = MDF(r"test.mdf")
    with Timer("Get all channels", f"asammdf {asammdf_version} mdfv3", fmt) as timer:
        for i, gp in enumerate(x.groups):
            gp = typing.cast(mdf_v3.Group | mdf_v4.Group, gp)
            for j in range(len(gp.channels)):
                x.get(group=i, index=j, samples_only=True)
    output.send([timer.output, timer.error])


def get_all_mdf4(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    x = MDF(r"test.mf4")
    with Timer("Get all channels", f"asammdf {asammdf_version} mdfv4", fmt) as timer:
        t = perf_counter()
        counter = 0
        to_break = False
        for i, gp in enumerate(x.groups):
            gp = typing.cast(mdf_v3.Group | mdf_v4.Group, gp)
            if to_break:
                break
            for j in range(len(gp.channels)):
                t2 = perf_counter()
                if t2 - t > 60:
                    timer.message += f" {counter / (t2 - t)}/s"
                    to_break = True
                    break
                x.get(group=i, index=j, samples_only=True)
                counter += 1
    output.send([timer.output, timer.error])


def get_all_mdf4_column(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    x = MDF(r"test_column.mf4")
    with Timer("Get all channels", f"asammdf {asammdf_version} column mdfv4", fmt) as timer:
        t = perf_counter()
        counter = 0
        to_break = False
        for i, gp in enumerate(x.groups):
            gp = typing.cast(mdf_v3.Group | mdf_v4.Group, gp)
            if to_break:
                break
            for j in range(len(gp.channels)):
                t2 = perf_counter()
                if t2 - t > 60:
                    timer.message += f" {counter / (t2 - t)}/s"
                    to_break = True
                    break
                x.get(group=i, index=j, samples_only=False)
                counter += 1
    output.send([timer.output, timer.error])


def convert_v3_v4(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    with MDF(r"test.mdf") as x:
        with Timer("Convert file", f"asammdf {asammdf_version} v3 to v4", fmt) as timer:
            x.convert("4.10")
    output.send([timer.output, timer.error])


def convert_v4_v410(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    with MDF(r"test.mf4") as x:
        with Timer("Convert file", f"asammdf {asammdf_version} v4 to v410", fmt) as timer:
            y = x.convert("4.10")
            y.close()
    output.send([timer.output, timer.error])


def convert_v4_v420(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    with MDF(r"test.mf4") as x:
        with Timer("Convert file", f"asammdf {asammdf_version} v4 to v420", fmt) as timer:
            y = x.convert("4.20")
            y.close()
    output.send([timer.output, timer.error])


def merge_v3(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    files = [r"test.mdf"] * 3
    with Timer("Merge 3 files", f"asammdf {asammdf_version} v3", fmt) as timer:
        MDF.concatenate(files, version="3.30")
    output.send([timer.output, timer.error])


def merge_v4(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    files = [r"test.mf4"] * 3

    with Timer("Merge 3 files", f"asammdf {asammdf_version} v4", fmt) as timer:
        MDF.concatenate(files, version="4.10")
    output.send([timer.output, timer.error])


#
# mdfreader
#


def open_reader3(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    with Timer("Open file", f"mdfreader {mdfreader_version} mdfv3", fmt) as timer:
        MDFreader(r"test.mdf")
    output.send([timer.output, timer.error])


def open_reader3_nodata(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    with Timer("Open file", f"mdfreader {mdfreader_version} no_data_loading mdfv3", fmt) as timer:
        MDFreader(r"test.mdf", no_data_loading=True)
    output.send([timer.output, timer.error])


def open_reader3_compression(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    with Timer("Open file", f"mdfreader {mdfreader_version} compress mdfv3", fmt) as timer:
        MDFreader(r"test.mdf", compression="blosc")
    output.send([timer.output, timer.error])


def open_reader4(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    with Timer("Open file", f"mdfreader {mdfreader_version} mdfv4", fmt) as timer:
        MDFreader(r"test.mf4")
    output.send([timer.output, timer.error])


def open_reader4_nodata(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    with Timer("Open file", f"mdfreader {mdfreader_version} no_data_loading mdfv4", fmt) as timer:
        MDFreader(r"test.mf4", no_data_loading=True)
    output.send([timer.output, timer.error])


def open_reader4_compression(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    with Timer("Open file", f"mdfreader {mdfreader_version} compress mdfv4", fmt) as timer:
        MDFreader(r"test.mf4", compression="blosc")
    output.send([timer.output, timer.error])


def save_reader3(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    x = MDFreader(r"test.mdf")
    with Timer("Save file", f"mdfreader {mdfreader_version} mdfv3", fmt) as timer:
        x.write(r"x.mdf")
    output.send([timer.output, timer.error])


def save_reader3_nodata(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    x = MDFreader(r"test.mdf", no_data_loading=True)
    with Timer("Save file", f"mdfreader {mdfreader_version} no_data_loading mdfv3", fmt) as timer:
        x.write(r"x.mdf")
    output.send([timer.output, timer.error])


def save_reader3_compression(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    with Timer("Save file", f"mdfreader {mdfreader_version} compress mdfv3", fmt) as outer_timer:
        x = MDFreader(r"test.mdf", compression="blosc")
        with Timer("Save file", f"mdfreader {mdfreader_version} compress mdfv3", fmt) as timer:
            x.write(r"x.mdf")
        output.send([timer.output, timer.error])
    if outer_timer.error:
        output.send([timer.output, timer.error])


def save_reader4(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    x = MDFreader(r"test.mf4")
    with Timer("Save file", f"mdfreader {mdfreader_version} mdfv4", fmt) as timer:
        x.write(r"x.mf4")
    output.send([timer.output, timer.error])


def save_reader4_nodata(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    x = MDFreader(r"test.mf4", no_data_loading=True)
    with Timer("Save file", f"mdfreader {mdfreader_version} no_data_loading mdfv4", fmt) as timer:
        x.write(r"x.mf4")
    output.send([timer.output, timer.error])


def save_reader4_compression(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    x = MDFreader(r"test.mf4", compression="blosc")
    with Timer("Save file", f"mdfreader {mdfreader_version} compress mdfv4", fmt) as timer:
        x.write(r"x.mf4")
    output.send([timer.output, timer.error])


def get_all_reader3(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    x = MDFreader(r"test.mdf")
    with Timer("Get all channels", f"mdfreader {mdfreader_version} mdfv3", fmt) as timer:
        for s in x:
            x.get_channel_data(s)
    output.send([timer.output, timer.error])


def get_all_reader3_nodata(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    x = MDFreader(r"test.mdf", no_data_loading=True)
    with Timer("Get all channels", f"mdfreader {mdfreader_version} nodata mdfv3", fmt) as timer:
        for s in x:
            x.get_channel_data(s)
    output.send([timer.output, timer.error])


def get_all_reader3_compression(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    x = MDFreader(r"test.mdf", compression="blosc")
    with Timer("Get all channels", f"mdfreader {mdfreader_version} compress mdfv3", fmt) as timer:
        for s in x:
            x.get_channel_data(s)

        with open("D:\\TMP\\f.txt", "w") as f:
            f.write("OK")
    output.send([timer.output, timer.error])


def get_all_reader4(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    x = MDFreader(r"test.mf4")
    with Timer("Get all channels", f"mdfreader {mdfreader_version} mdfv4", fmt) as timer:
        t = perf_counter()
        counter = 0
        to_break = False
        for s in x:
            t2 = perf_counter()
            if t2 - t > 60:
                timer.message += f" {counter / (t2 - t)}/s"
                to_break = True
                break
            x.get_channel_data(s)
            counter += 1
    output.send([timer.output, timer.error])


def get_all_reader4_nodata(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    x = MDFreader(r"test.mf4", no_data_loading=True)
    with Timer("Get all channels", f"mdfreader {mdfreader_version} nodata mdfv4", fmt) as timer:
        t = perf_counter()
        counter = 0
        to_break = False
        for s in x:
            t2 = perf_counter()
            if t2 - t > 60:
                timer.message += f" {counter / (t2 - t)}/s"
                to_break = True
                break
            x.get_channel_data(s)
            counter += 1
    output.send([timer.output, timer.error])


def get_all_reader4_compression(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    x = MDFreader(r"test.mf4", compression="blosc")
    with Timer("Get all channels", f"mdfreader {mdfreader_version} compress mdfv4", fmt) as timer:
        t = perf_counter()
        counter = 0
        to_break = False
        for s in x:
            t2 = perf_counter()
            if t2 - t > 60:
                timer.message += f" {counter / (t2 - t)}/s"
                to_break = True
                break
            x.get_channel_data(s)
            counter += 1
    output.send([timer.output, timer.error])


def merge_reader_v3(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    files = [r"test.mdf"] * 3
    with Timer("Merge 3 files", f"mdfreader {mdfreader_version} v3", fmt) as timer:
        x1 = MDFreader(files[0])
        x1.resample(0.01)
        x2 = MDFreader(files[1])
        x2.resample(0.01)
        x1.merge_mdf(x2)
        x2 = MDFreader(files[2])
        x2.resample(0.01)
        x1.merge_mdf(x2)
    output.send([timer.output, timer.error])


def merge_reader_v3_compress(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    files = [r"test.mdf"] * 3
    with Timer("Merge 3 files", f"mdfreader {mdfreader_version} compress v3", fmt) as timer:
        x1 = MDFreader(files[0], compression="blosc")
        x1.resample(0.01)
        x2 = MDFreader(files[1], compression="blosc")
        x2.resample(0.01)
        x1.merge_mdf(x2)
        x2 = MDFreader(files[2], compression="blosc")
        x2.resample(0.01)
        x1.merge_mdf(x2)
    output.send([timer.output, timer.error])


def merge_reader_v3_nodata(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    files = [r"test.mdf"] * 3
    with Timer("Merge 3 files", f"mdfreader {mdfreader_version} nodata v3", fmt) as timer:
        x1 = MDFreader(files[0], no_data_loading=True)
        x1.resample(0.01)
        x2 = MDFreader(files[1], no_data_loading=True)
        x2.resample(0.01)
        x1.merge_mdf(x2)
        x2 = MDFreader(files[2], no_data_loading=True)
        x2.resample(0.01)
        x1.merge_mdf(x2)
    output.send([timer.output, timer.error])


def merge_reader_v4(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    files = [r"test.mf4"] * 3

    with Timer("Merge 3 files", f"mdfreader {mdfreader_version} v4", fmt) as timer:
        x1 = MDFreader(files[0])
        x1.resample(0.01)
        x2 = MDFreader(files[1])
        x2.resample(0.01)
        x1.merge_mdf(x2)
        x2 = MDFreader(files[2])
        x2.resample(0.01)
        x1.merge_mdf(x2)

    output.send([timer.output, timer.error])


def merge_reader_v4_compress(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    files = [r"test.mf4"] * 3
    with Timer("Merge 3 files", f"mdfreader {mdfreader_version} compress v4", fmt) as timer:
        x1 = MDFreader(files[0], compression="blosc")
        x1.resample(0.01)
        x2 = MDFreader(files[1], compression="blosc")
        x2.resample(0.01)
        x1.merge_mdf(x2)
        x2 = MDFreader(files[2], compression="blosc")
        x2.resample(0.01)
        x1.merge_mdf(x2)

    output.send([timer.output, timer.error])


def merge_reader_v4_nodata(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    files = [r"test.mf4"] * 3
    with Timer("Merge 3 files", f"mdfreader {mdfreader_version} nodata v4", fmt) as timer:
        x1 = MDFreader(files[0], no_data_loading=True)
        x1.resample(0.01)
        x2 = MDFreader(files[1], no_data_loading=True)
        x2.resample(0.01)
        x1.merge_mdf(x2)
        x2 = MDFreader(files[2], no_data_loading=True)
        x2.resample(0.01)
        x1.merge_mdf(x2)

    output.send([timer.output, timer.error])


#
# utility functions
#


def filter_asam(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    with Timer("Filter file", f"asammdf {asammdf_version} mdfv4", fmt) as timer:
        x = MDF(r"test.mf4").filter([(None, i, int(f"{j}5")) for i in range(10, 20) for j in range(1, 20)])
        t = perf_counter()
        counter = 0
        to_break = False
        for i, gp in enumerate(x.groups):
            gp = typing.cast(mdf_v3.Group | mdf_v4.Group, gp)
            if to_break:
                break
            for j in range(len(gp.channels)):
                t2 = perf_counter()
                if t2 - t > 60:
                    timer.message += f" {counter / (t2 - t)}/s"
                    to_break = True
                    break
                x.get(group=i, index=j, samples_only=True)
                counter += 1
    output.send([timer.output, timer.error])


def filter_reader4(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    with Timer("Filter file", f"mdfreader {mdfreader_version} mdfv4", fmt) as timer:
        x = MDFreader(
            r"test.mf4",
            channel_list=[f"Channel_{i}_{j}5" for i in range(10) for j in range(1, 20)],
        )
        t = perf_counter()
        counter = 0
        for s in x:
            t2 = perf_counter()
            if t2 - t > 60:
                timer.message += f" {counter / (t2 - t)}/s"
                break
            x.get_channel_data(s)
            counter += 1
    output.send([timer.output, timer.error])


def filter_reader4_compression(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    with Timer("Filter file", f"mdfreader {mdfreader_version} compression mdfv4", fmt) as timer:
        x = MDFreader(
            r"test.mf4",
            compression="blosc",
            channel_list=[f"Channel_{i}_{j}5" for i in range(10) for j in range(1, 20)],
        )
        t = perf_counter()
        counter = 0
        for s in x:
            t2 = perf_counter()
            if t2 - t > 60:
                timer.message += f" {counter / (t2 - t)}/s"
                break
            x.get_channel_data(s)
            counter += 1
    output.send([timer.output, timer.error])


def filter_reader4_nodata(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    with Timer("Filter file", f"mdfreader {mdfreader_version} nodata mdfv4", fmt) as timer:
        x = MDFreader(
            r"test.mf4",
            no_data_loading=True,
            channel_list=[f"Channel_{i}_{j}5" for i in range(10) for j in range(1, 20)],
        )
        t = perf_counter()
        counter = 0
        for s in x:
            t2 = perf_counter()
            if t2 - t > 60:
                timer.message += f" {counter / (t2 - t)}/s"
                break
            x.get_channel_data(s)
            counter += 1
    output.send([timer.output, timer.error])


def cut_asam(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    x = MDF(r"test.mf4")
    t = x.get_master(0)
    start, stop = 0.2 * (t[-1] - t[0]) + t[0], 0.8 * (t[-1] - t[0]) + t[0]
    with Timer("Cut file", f"asammdf {asammdf_version} mdfv4", fmt) as timer:
        x = x.cut(start=start, stop=stop)

    output.send([timer.output, timer.error])


def cut_reader4(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    x = MDFreader(r"test.mf4")
    t = x.get_channel_data(list(x.masterChannelList)[0])
    begin, end = 0.2 * (t[-1] - t[0]) + t[0], 0.8 * (t[-1] - t[0]) + t[0]
    with Timer("Cut file", f"mdfreader {mdfreader_version} mdfv4", fmt) as timer:
        x.cut(begin=begin, end=end)
    output.send([timer.output, timer.error])


def cut_reader4_compression(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    x = MDFreader(r"test.mf4", compression="blosc")
    t = x.get_channel_data(list(x.masterChannelList)[0])
    begin, end = 0.2 * (t[-1] - t[0]) + t[0], 0.8 * (t[-1] - t[0]) + t[0]
    with Timer("Cut file", f"mdfreader {mdfreader_version} compression mdfv4", fmt) as timer:
        x.cut(begin=begin, end=end)
    output.send([timer.output, timer.error])


def cut_reader4_nodata(output: Connection[object, object], fmt: Literal["rst", "md"]) -> None:
    x = MDFreader(r"test.mf4", no_data_loading=True)
    t = x.get_channel_data(list(x.masterChannelList)[0])
    begin, end = 0.2 * (t[-1] - t[0]) + t[0], 0.8 * (t[-1] - t[0]) + t[0]
    with Timer("Cut file", f"mdfreader {mdfreader_version} nodata mdfv4", fmt) as timer:
        x.cut(begin=begin, end=end)
    output.send([timer.output, timer.error])


def table_header(topic: str, fmt: Literal["rst", "md"] = "rst") -> list[str]:
    output = []
    if fmt == "rst":
        result = "{:<50} {:>9} {:>8}".format(topic, "Time [ms]", "RAM [MB]")
        output.append("")
        output.append("{} {} {}".format("=" * 50, "=" * 9, "=" * 8))
        output.append(result)
        output.append("{} {} {}".format("=" * 50, "=" * 9, "=" * 8))
    elif fmt == "md":
        result = "|{:<50}|{:>9}|{:>8}|".format(topic, "Time [ms]", "RAM [MB]")
        output.append("")
        output.append(result)
        output.append("|{}|{}|{}|".format("-" * 50, "-" * 9, "-" * 8))
    return output


def table_end(fmt: Literal["rst", "md"] = "rst") -> list[str]:
    if fmt == "rst":
        return ["{} {} {}".format("=" * 50, "=" * 9, "=" * 8), ""]
    elif fmt == "md":
        return [
            "",
        ]


def main(text_output: bool, fmt: Literal["rst", "md"]) -> None:
    if os.path.dirname(__file__):
        os.chdir(os.path.dirname(__file__))
    for version in ("3.30", "4.10"):
        generate_test_files(version)

    mdf = MDF("test.mdf", "minimum")
    v3_size = os.path.getsize("test.mdf") // 1024 // 1024
    v3_groups = len(mdf.groups)
    v3_channels = sum(len(gp.channels) for gp in mdf.groups)
    v3_version = mdf.version

    mdf = MDF("test.mf4", "minimum")
    mdf.get_master(0)
    v4_size = os.path.getsize("test.mf4") // 1024 // 1024
    v4_groups = len(mdf.groups)
    v4_channels = sum(len(gp.channels) for gp in mdf.groups)
    v4_version = mdf.version

    listen, send = multiprocessing.Pipe()
    output = MyList()
    errors: list[str] = []

    installed_ram = round(psutil.virtual_memory().total / 1024 / 1024 / 1024)

    output.append("\n\nBenchmark environment\n")
    output.append(f"* {sys.version}")
    output.append(f"* {platform.platform()}")
    output.append(f"* {platform.processor()}")
    output.append(f"* numpy {np.__version__}")
    output.append(f"* {installed_ram}GB installed RAM\n")
    output.append("Notations used in the results\n")
    output.append("* compress = mdfreader mdf object created with compression=blosc")
    output.append("* nodata = mdfreader mdf object read with no_data_loading=True")
    output.append("\nFiles used for benchmark:\n")
    output.append(f"* mdf version {v3_version}")
    output.append(f"    * {v3_size} MB file size")
    output.append(f"    * {v3_groups} groups")
    output.append(f"    * {v3_channels} channels")
    output.append(f"* mdf version {v4_version}")
    output.append(f"    * {v4_size} MB file size")
    output.append(f"    * {v4_groups} groups")
    output.append(f"    * {v4_channels} channels\n\n")

    OPEN, SAVE, GET, CONVERT, MERGE, FILTER, CUT = 1, 1, 1, 1, 1, 1, 1

    tests: tuple[Callable[[Connection[object, object], Literal["rst", "md"]], None], ...] = (
        open_mdf3,
        # open_reader3,
        # open_reader3_nodata,
        # open_reader3_compression,
        open_mdf4,
        # open_mdf4_column,
        # open_reader4,
        # open_reader4_nodata,
        # open_reader4_compression,
    )

    if tests and OPEN:
        output.extend(table_header("Open file", fmt))
        for func in tests:
            thr = multiprocessing.Process(target=func, args=(send, fmt))
            thr.start()
            thr.join()
            result, err = listen.recv()
            output.append(result)
            errors.append(err)
        output.extend(table_end(fmt))

    tests = (
        save_mdf3,
        # save_reader3,
        # save_reader3_nodata,
        # save_reader3_compression,
        save_mdf4,
        # save_mdf4_column,
        # save_reader4,
        # save_reader4_nodata,
        # save_reader4_compression,
    )

    if tests and SAVE:
        output.extend(table_header("Save file", fmt))
        for func in tests:
            thr = multiprocessing.Process(target=func, args=(send, fmt))
            thr.start()
            thr.join()
            result, err = listen.recv()
            output.append(result)
            errors.append(err)
        output.extend(table_end(fmt))

    tests = (
        get_all_mdf3,
        # get_all_reader3,
        # get_all_reader3_nodata,
        # get_all_reader3_compression,
        get_all_mdf4,
        # get_all_mdf4_column,
        # get_all_reader4,
        # get_all_reader4_nodata,
        # get_all_reader4_compression,
    )

    if tests and GET:
        output.extend(table_header("Get all channels (36424 calls)", fmt))
        for func in tests:
            thr = multiprocessing.Process(target=func, args=(send, fmt))
            thr.start()
            thr.join()
            result, err = listen.recv()
            output.append(result)
            errors.append(err)
        output.extend(table_end(fmt))

    tests = (
        convert_v3_v4,
        convert_v4_v410,
        convert_v4_v420,
    )

    if tests and CONVERT:
        output.extend(table_header("Convert file", fmt))
        for func in tests:
            thr = multiprocessing.Process(target=func, args=(send, fmt))
            thr.start()
            thr.join()
            result, err = listen.recv()
            output.append(result)
            errors.append(err)
        output.extend(table_end(fmt))

    tests = (
        merge_v3,
        # merge_reader_v3,
        # merge_reader_v3_nodata,
        # merge_reader_v3_compress,
        merge_v4,
        # merge_reader_v4,
        # merge_reader_v4_nodata,
        # merge_reader_v4_compress,
    )

    if tests and MERGE:
        output.extend(table_header("Merge 3 files", fmt))
        for func in tests:
            thr = multiprocessing.Process(target=func, args=(send, fmt))
            thr.start()
            thr.join()
            result, err = listen.recv()
            output.append(result)
            errors.append(err)
        output.extend(table_end(fmt))

    tests = (
        filter_asam,
        # filter_reader4,
        # filter_reader4_compression,
        # filter_reader4_nodata,
    )

    if tests and FILTER:
        output.extend(table_header("Filter 200 channels", fmt))
        for func in tests:
            thr = multiprocessing.Process(target=func, args=(send, fmt))
            thr.start()
            thr.join()
            result, err = listen.recv()
            output.append(result)
            errors.append(err)
        output.extend(table_end(fmt))

    tests = (
        cut_asam,
        # cut_reader4,
        # cut_reader4_compression,
        # cut_reader4_nodata,
    )

    if tests and CUT:
        output.extend(table_header("Cut file from 20% to 80%", fmt))
        for func in tests:
            thr = multiprocessing.Process(target=func, args=(send, fmt))
            thr.start()
            thr.join()
            result, err = listen.recv()
            output.append(result)
            errors.append(err)
        output.extend(table_end(fmt))

    errors = [err for err in errors if err]
    if errors:
        print("\n\nERRORS\n", "\n".join(errors))

    if text_output:
        arch = "x86" if platform.architecture()[0] == "32bit" else "x64"
        file = f"{arch}_asammdf_{asammdf_version}_mdfreader_{mdfreader_version}.{fmt}"
        with open(file, "w") as out:
            out.write("\n".join(output))

    for file in ("x.mdf", "x.mf4"):
        if PYVERSION >= 3:
            try:
                os.remove(file)
            except FileNotFoundError:
                pass
        else:
            try:
                os.remove(file)
            except OSError:
                pass


def _cmd_line_parser() -> argparse.ArgumentParser:
    """
    return a command line parser. It is used when generating the documentation
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        help=("path to test files, if not provided the script folder is used"),
    )
    parser.add_argument(
        "--text_output",
        action="store_true",
        help="option to save the results to text file",
    )
    parser.add_argument(
        "--format",
        default="rst",
        nargs="?",
        choices=["rst", "md"],
        help="text formatting",
    )

    return parser


if __name__ == "__main__":
    cmd_parser = _cmd_line_parser()
    args = cmd_parser.parse_args(sys.argv[1:])

    main(args.text_output, args.format)
#
#    x = MDF(r"test_column.mf4")
#    with Timer("Get all channels", f"asammdf {asammdf_version} column mdfv4", "rst") as timer:
#        t = perf_counter()
#        counter = 0
#        to_break = False
#        for i, gp in enumerate(x.groups):
#            gp = typing.cast(mdf_v3.Group | mdf_v4.Group, gp)
#            if to_break:
#                break
#            for j in range(len(gp.channels)):
#                t2 = perf_counter()
#                if t2 - t > 60:
#                    timer.message += " {}/s".format(counter / (t2 - t))
#                    to_break = True
#                    break
#                x.get(group=i, index=j, samples_only=True)
#                counter += 1
#
#    print(timer.output, timer.error)

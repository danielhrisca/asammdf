# -*- coding: utf-8 -*-
from datetime import datetime
from functools import reduce
from io import StringIO
import json
from pathlib import Path
import re
from struct import unpack
from threading import Thread
from time import sleep
import traceback

import lxml
import natsort
from numexpr import evaluate
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from ..blocks.conversion_utils import from_dict
from ..mdf import MDF, MDF2, MDF3, MDF4
from ..signal import Signal
from .dialogs.error_dialog import ErrorDialog

ERROR_ICON = None
NO_ERROR_ICON = None


COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

COMPARISON_NAME = re.compile(r"(\s*\d+:)?(?P<name>.+)")
SIG_RE = re.compile(r"\{\{(?!\}\})(?P<name>.*?)\}\}")

TERMINATED = object()


def excepthook(exc_type, exc_value, tracebackobj):
    """
    Global function to catch unhandled exceptions.

    Parameters
    ----------
    exc_type : str
        exception type
    exc_value : int
        exception value
    tracebackobj : traceback
        traceback object
    """
    separator = "-" * 80
    notice = "The following error was triggered:"

    now = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")

    info = StringIO()
    traceback.print_tb(tracebackobj, None, info)
    info.seek(0)
    info = info.read()

    errmsg = f"{exc_type}\t \n{exc_value}"
    sections = [now, separator, errmsg, separator, info]
    msg = "\n".join(sections)

    print("".join(traceback.format_tb(tracebackobj)))
    print("{0}: {1}".format(exc_type, exc_value))

    ErrorDialog(
        message=errmsg, trace=msg, title="The following error was triggered"
    ).exec_()


def extract_mime_names(data):
    def fix_comparison_name(data):
        for i, (
            name,
            group_index,
            channel_index,
            mdf_uuid,
            item_type,
            ranges,
        ) in enumerate(data):
            if item_type == "channel":
                if (group_index, channel_index) != (-1, -1):
                    name = COMPARISON_NAME.match(name).group("name").strip()
                    data[i][0] = name
            else:
                fix_comparison_name(channel_index)

    names = []
    if data.hasFormat("application/octet-stream-asammdf"):
        data = bytes(data.data("application/octet-stream-asammdf")).decode("utf-8")
        data = json.loads(data)
        fix_comparison_name(data)
        names = data

    return names


def load_dsp(file):
    def parse_channels(display):
        channels = []
        for elem in display.iterchildren():
            if elem.tag == "CHANNEL":
                color_ = int(elem.get("color"))
                c = 0
                for i in range(3):
                    c = c << 8
                    c += color_ & 0xFF
                    color_ = color_ >> 8

                if c in (0xFFFFFF, 0x0):
                    c = 0x808080

                gain = float(elem.get("gain"))
                offset = float(elem.get("offset")) / 100

                channels.append(
                    {
                        "color": f"#{c:06X}",
                        "common_axis": False,
                        "computed": False,
                        "enabled": elem.get("on") == "1",
                        "fmt": "{}",
                        "individual_axis": False,
                        "name": elem.get("name"),
                        "precision": 3,
                        "ranges": [],
                        "unit": "",
                        "type": "channel",
                        "y_range": [
                            -gain * offset,
                            -gain * offset + 19 * gain,
                        ],
                    }
                )

            elif elem.tag.startswith("GROUP"):
                channels.append(
                    {
                        "name": elem.get("data"),
                        "enabled": elem.get("on") == "1",
                        "type": "group",
                        "channels": parse_channels(elem),
                        "pattern": None,
                    }
                )

            elif elem.tag == "CHANNEL_PATTERN":

                try:
                    info = {
                        "pattern": elem.get("name_pattern"),
                        "name": elem.get("name_pattern"),
                        "match_type": "Wildcard",
                        "filter_type": elem.get("filter_type"),
                        "filter_value": float(elem.get("filter_value")),
                        "raw": bool(int(elem.get("filter_use_raw"))),
                    }

                    multi_color = elem.find("MULTI_COLOR")

                    ranges = {}

                    if multi_color is not None:
                        for color in multi_color.findall("color"):
                            min_ = float(color.find("min").get("data"))
                            max_ = float(color.find("max").get("data"))
                            color_ = int(color.find("color").get("data"))
                            c = 0
                            for i in range(3):
                                c = c << 8
                                c += color_ & 0xFF
                                color_ = color_ >> 8
                            color = QtGui.QColor(f"#{c:06X}")
                            ranges.append(
                                {
                                    "background_color": color,
                                    "font_color": color,
                                    "op1": "<=",
                                    "op2": "<=",
                                    "value1": min_,
                                    "value2": max_,
                                }
                            )

                    info["ranges"] = ranges

                    channels.append(
                        {
                            "channels": [],
                            "enabled": True,
                            "name": info["pattern"],
                            "pattern": info,
                            "type": "group",
                            "ranges": [],
                        }
                    )

                except:
                    continue

        return channels

    def parse_virtual_channels(display):
        channels = {}

        if display is None:
            return channels

        for item in display.findall("V_CHAN"):
            try:
                virtual_channel = {}

                parent = item.find("VIR_TIME_CHAN")
                vtab = item.find("COMPU_VTAB")
                if parent is None or vtab is None:
                    continue

                name = item.get("name")

                virtual_channel["name"] = name
                virtual_channel["parent"] = parent.get("data")
                virtual_channel["comment"] = item.find("description").get("data")

                conv = {}
                for i, item in enumerate(vtab.findall("tab")):
                    conv[f"val_{i}"] = float(item.get("min"))
                    conv[f"text_{i}"] = item.get("text")

                virtual_channel["vtab"] = conv

                channels[name] = virtual_channel
            except:
                continue

        return channels

    dsp = Path(file).read_bytes().replace(b"\0", b"")
    dsp = lxml.etree.fromstring(dsp)

    channels = parse_channels(dsp.find("DISPLAY_INFO"))

    info = {}
    info["selected_channels"] = []

    info["windows"] = windows = []

    if channels:

        plot = {
            "type": "Plot",
            "title": "Display channels",
            "configuration": {
                "channels": channels,
            },
        }

        windows.append(plot)

    channels = parse_virtual_channels(dsp.find("VIRTUAL_CHANNEL"))

    if channels:
        plot = {
            "type": "Plot",
            "title": "Display channels",
            "configuration": {
                "channels": [
                    {
                        "color": COLORS[i % len(COLORS)],
                        "common_axis": False,
                        "computed": True,
                        "computation": {
                            "type": "expression",
                            "expression": "{{" + ch["parent"] + "}}",
                        },
                        "enabled": True,
                        "fmt": "{}",
                        "individual_axis": False,
                        "name": ch["parent"],
                        "precision": 3,
                        "ranges": [],
                        "unit": "",
                        "conversion": ch["vtab"],
                        "user_defined_name": ch["name"],
                    }
                    for i, ch in enumerate(channels.values())
                ]
            },
        }

        windows.append(plot)

    return info


def load_lab(file):
    sections = {}
    with open(file, "r") as lab:
        for line in lab:
            line = line.strip()
            if not line:
                continue

            if line.startswith("[") and line.endswith("]"):
                section_name = line.strip("[]")
                s = []
                sections[section_name] = s

            else:
                s.append(line)

    return {name: channels for name, channels in sections.items() if channels}


def run_thread_with_progress(
    widget, target, kwargs, factor=100, offset=0, progress=None
):
    termination_request = False

    thr = WorkerThread(target=target, kwargs=kwargs)

    thr.start()

    while widget.progress is None and thr.is_alive():
        sleep(0.1)

    while thr.is_alive():
        termination_request = progress.wasCanceled()
        if termination_request:
            MDF._terminate = True
            MDF2._terminate = True
            MDF3._terminate = True
            MDF4._terminate = True
        else:
            if widget.progress is not None:
                if widget.progress != (0, 0):
                    progress.setValue(
                        int(widget.progress[0] / widget.progress[1] * factor) + offset
                    )
                else:
                    progress.setRange(0, 0)
        QtCore.QCoreApplication.processEvents()
        sleep(0.1)

    if termination_request:
        MDF._terminate = False
        MDF2._terminate = False
        MDF3._terminate = False
        MDF4._terminate = False

    progress.setValue(factor + offset)

    if thr.error:
        widget.progress = None
        progress.cancel()
        raise Exception(thr.error)

    widget.progress = None

    if termination_request:
        return TERMINATED
    else:
        return thr.output


def setup_progress(parent, title, message, icon_name):
    progress = QtWidgets.QProgressDialog(message, "", 0, 100, parent)

    progress.setWindowModality(QtCore.Qt.ApplicationModal)
    progress.setCancelButton(None)
    progress.setAutoClose(True)
    progress.setWindowTitle(title)
    icon = QtGui.QIcon()
    icon.addPixmap(
        QtGui.QPixmap(f":/{icon_name}.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
    )
    progress.setWindowIcon(icon)
    progress.setMinimumWidth(600)
    progress.show()

    return progress


class WorkerThread(Thread):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.output = None
        self.error = ""

    def run(self):
        try:
            self.output = self._target(*self._args, **self._kwargs)
        except:
            self.error = traceback.format_exc()


def compute_signal(description, measured_signals, all_timebase):
    type_ = description["type"]

    if type_ == "arithmetic":
        op = description["op"]

        operand1 = description["operand1"]
        if isinstance(operand1, dict):
            operand1 = compute_signal(operand1, measured_signals, all_timebase)
        elif isinstance(operand1, str):
            operand1 = measured_signals[operand1]

        operand2 = description["operand2"]
        if isinstance(operand2, dict):
            operand2 = compute_signal(operand2, measured_signals, all_timebase)
        elif isinstance(operand2, str):
            operand2 = measured_signals[operand2]

        result = eval(f"operand1 {op} operand2")
        if not hasattr(result, "name"):
            result = Signal(
                name="_",
                samples=np.ones(len(all_timebase)) * result,
                timestamps=all_timebase,
            )

    elif type_ == "function":
        function = description["name"]
        args = description["args"]

        channel = description["channel"]

        if isinstance(channel, dict):
            channel = compute_signal(channel, measured_signals, all_timebase)
        else:
            channel = measured_signals[channel]

        func = getattr(np, function)

        if function in [
            "arccos",
            "arcsin",
            "arctan",
            "cos",
            "deg2rad",
            "degrees",
            "rad2deg",
            "radians",
            "sin",
            "tan",
            "floor",
            "rint",
            "fix",
            "trunc",
            "cumprod",
            "cumsum",
            "diff",
            "exp",
            "log10",
            "log",
            "log2",
            "absolute",
            "cbrt",
            "sqrt",
            "square",
            "gradient",
        ]:

            samples = func(channel.samples)
            if function == "diff":
                timestamps = channel.timestamps[1:]
            else:
                timestamps = channel.timestamps

        elif function == "around":
            samples = func(channel.samples, *args)
            timestamps = channel.timestamps
        elif function == "clip":
            samples = func(channel.samples, *args)
            timestamps = channel.timestamps

        result = Signal(samples=samples, timestamps=timestamps, name="_")

    elif type_ == "expression":
        expression_string = description["expression"]
        expression_string = "".join(expression_string.splitlines())
        names = [match.group("name") for match in SIG_RE.finditer(expression_string)]
        positions = [
            (i, match.start(), match.end())
            for i, match in enumerate(SIG_RE.finditer(expression_string))
        ]
        positions.reverse()

        expression = expression_string
        for idx, start, end in positions:
            expression = expression[:start] + f"X_{idx}" + expression[end:]

        signals = [measured_signals[name] for name in names]
        common_timebase = reduce(np.union1d, [sig.timestamps for sig in signals])
        signals = {
            f"X_{i}": sig.interp(common_timebase).samples
            for i, sig in enumerate(signals)
        }

        samples = evaluate(expression, local_dict=signals)

        result = Signal(
            name="_",
            samples=samples,
            timestamps=common_timebase,
        )

    return result


class HelperChannel:

    __slots__ = "entry", "name", "added"

    def __init__(self, entry, name):
        self.name = name
        self.entry = entry
        self.added = False


def copy_ranges(ranges):
    if ranges:
        new_ranges = []
        for range_info in ranges:
            range_info = dict(range_info)
            for color_name in ("background_color", "font_color"):
                color = range_info[color_name]
                if isinstance(color, QtGui.QBrush):
                    range_info[color_name] = QtGui.QBrush(color)
                elif isinstance(color, QtGui.QColor):
                    range_info[color_name] = QtGui.QColor(color)
            new_ranges.append(range_info)

        return new_ranges
    else:
        return ranges


def get_colors_using_ranges(
    value, ranges, default_background_color, default_font_color
):
    new_background_color = default_background_color
    new_font_color = default_font_color

    if value is None:
        return new_background_color, new_font_color

    if ranges:

        for base_class in (float, str, int):
            if isinstance(value, base_class):
                for range_info in ranges:
                    (
                        background_color,
                        font_color,
                        op1,
                        op2,
                        value1,
                        value2,
                    ) = range_info.values()

                    result = False

                    if isinstance(value1, base_class):
                        if op1 == "==":
                            result = value1 == value
                        elif op1 == "!=":
                            result = value1 != value
                        elif op1 == "<=":
                            result = value1 <= value
                        elif op1 == "<":
                            result = value1 < value
                        elif op1 == ">=":
                            result = value1 >= value
                        elif op1 == ">":
                            result = value1 > value

                        if not result:
                            continue

                    if isinstance(value2, base_class):
                        if op2 == "==":
                            result = value == value2
                        elif op2 == "!=":
                            result = value != value2
                        elif op2 == "<=":
                            result = value <= value2
                        elif op2 == "<":
                            result = value < value2
                        elif op2 == ">=":
                            result = value >= value2
                        elif op2 == ">":
                            result = value > value2

                        if not result:
                            continue

                    if result:

                        new_background_color = background_color
                        new_font_color = font_color
                        break
                break

    return new_background_color, new_font_color


if __name__ == "__main__":
    pass

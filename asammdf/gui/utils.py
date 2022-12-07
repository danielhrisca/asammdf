# -*- coding: utf-8 -*-
from collections import defaultdict
from datetime import datetime
from functools import reduce
import inspect
from io import StringIO
import json
import math
import os
from pathlib import Path
import re
from textwrap import dedent, indent
from threading import Thread
from time import perf_counter, sleep
import traceback
from traceback import format_exc

import lxml
import natsort
from numexpr import evaluate
import numpy as np
import pandas as pd
from pyqtgraph import functions as fn
from PySide6 import QtCore, QtGui, QtWidgets

from ..blocks.options import FloatInterpolation, IntegerInterpolation
from ..mdf import MDF, MDF2, MDF3, MDF4
from ..signal import Signal
from .dialogs.error_dialog import ErrorDialog

ERROR_ICON = None
RANGE_INDICATOR_ICON = None
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

SCROLLBAR_STYLE = """
QTreeWidget {{ font-size: {font_size}pt; }}

QScrollBar:vertical {{
    border: 1px solid #61b2e2;
    background: {background};
    width: 14px;
    margin: 14px 0px 14px 0px;
}}
QScrollBar::handle:vertical {{
    border: 1px solid  {background};
    background: #61b2e2;
    min-height: 40px;
    width: 6px;
}}

QScrollBar::add-line:vertical {{
    border: 1px solid #61b2e2;
    background: {background};
    height: 13px;
    subcontrol-position: bottom;
    subcontrol-origin: margin;
}}
QScrollBar::sub-line:vertical {{
    border: 1px solid #61b2e2;
    background: {background};
    height: 13px;
    subcontrol-position: top;
    subcontrol-origin: margin;
}}

QScrollBar:up-arrow:vertical {{
    image: url(:up.png);
    width: 13px;
    height: 13px;
}}
QScrollBar:down-arrow:vertical {{
    image: url(:down.png);
    width: 13px;
    height: 13px;
}}


QScrollBar:horizontal {{
    border: 1px solid #61b2e2;
    background: {background};
    height: 14px;
    margin: 0px 14px 0px 14px;
}}
QScrollBar::handle:horizontal {{
    border: 1px solid  {background};
    background: #61b2e2;
    min-width: 40px;
    height: 14px;
}}
QScrollBar::add-line:horizontal {{
    border: 1px solid #61b2e2;
    background: {background};
    width: 13px;
    subcontrol-position: right;
    subcontrol-origin: margin;
}}
QScrollBar::sub-line:horizontal {{
    border: 1px solid #61b2e2;
    background: {background};
    width: 13 px;
    subcontrol-position: left;
    subcontrol-origin: margin;
}}

QScrollBar:left-arrow:horizontal {{
    image: url(:left2.png);
    width: 13px;
    height: 13px;
}}
QScrollBar:right-arrow:horizontal {{
    image: url(:right2.png);
    width: 13px;
    height: 13px;
}}"""

COMPARISON_NAME = re.compile(r"(\s*\d+:)?(?P<name>.+)")
SIG_RE = re.compile(r"\{\{(?!\}\})(?P<name>.*?)\}\}")

TERMINATED = object()

FONT_SIZE = [6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22, 24, 26, 28, 36, 48, 72]
VARIABLE = re.compile(r"(?P<var>\{\{[^}]+\}\})")
VARIABLE_GET_DATA = re.compile(r"get_data\s*\(\s*\"(?P<var>[^\"]+)")
C_FUNCTION = re.compile(r"\s+(?P<function>\S+)\s*\(\s*struct\s+DATA\s+\*data\s*\)")
FUNC_NAME = re.compile(r"def\s+(?P<name>\S+)\s*\(")


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
        for item in data:
            if item["type"] == "channel":
                if (item["group_index"], item["channel_index"]) != (-1, -1):
                    name = COMPARISON_NAME.match(item["name"]).group("name").strip()
                    item["name"] = name
            else:
                fix_comparison_name(item["channels"])

    names = []
    if data.hasFormat("application/octet-stream-asammdf"):
        data = bytes(data.data("application/octet-stream-asammdf")).decode("utf-8")
        data = json.loads(data)
        fix_comparison_name(data)
        names = data

    return names


def load_dsp(file, background="#000000", flat=False):
    if isinstance(background, str):
        background = fn.mkColor(background)

    def parse_channels(display):
        channels = []
        for elem in display.iterchildren():
            if elem.tag == "CHANNEL":
                channel_name = elem.get("name")

                comment = elem.find("COMMENT")
                if comment is not None:
                    comment = elem.get("text")
                else:
                    comment = ""

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

                multi_color = elem.find("MULTI_COLOR")

                ranges = []

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
                        color = fn.mkColor(f"#{c:06X}")
                        ranges.append(
                            {
                                "background_color": background,
                                "font_color": color,
                                "op1": "<=",
                                "op2": "<=",
                                "value1": min_,
                                "value2": max_,
                            }
                        )

                channels.append(
                    {
                        "color": f"#{c:06X}",
                        "common_axis": False,
                        "computed": False,
                        "flags": 0,
                        "comment": comment,
                        "enabled": elem.get("on") == "1",
                        "fmt": "{}",
                        "individual_axis": False,
                        "name": channel_name,
                        "mode": "phys",
                        "precision": 3,
                        "ranges": ranges,
                        "unit": "",
                        "type": "channel",
                        "y_range": [
                            -gain * offset,
                            -gain * offset + 19 * gain,
                        ],
                        "origin_uuid": "000000000000",
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
                        "origin_uuid": "000000000000",
                        "ranges": [],
                    }
                )

            elif elem.tag == "CHANNEL_PATTERN":

                try:
                    filter_type = elem.get("filter_type")
                    if filter_type == "None":
                        filter_type = "Unspecified"
                    info = {
                        "pattern": elem.get("name_pattern"),
                        "name": elem.get("name_pattern"),
                        "match_type": "Wildcard",
                        "filter_type": filter_type,
                        "filter_value": float(elem.get("filter_value")),
                        "raw": bool(int(elem.get("filter_use_raw"))),
                    }

                    multi_color = elem.find("MULTI_COLOR")

                    ranges = []

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
                            color = fn.mkColor(f"#{c:06X}")
                            ranges.append(
                                {
                                    "background_color": background,
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
                            "origin_uuid": "000000000000",
                        }
                    )

                except:
                    print(format_exc())
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
                    text = item.get("text")
                    if isinstance(text, bytes):
                        text = text.decode("utf-8", errors="replace")
                    conv[f"text_{i}"] = text

                virtual_channel["vtab"] = conv

                channels[name] = virtual_channel
            except:
                continue

        return channels

    def parse_c_functions(display):
        c_functions = set()

        if display is None:
            return c_functions

        for item in display.findall("CALC_FUNC"):
            string = item.text

            for match in C_FUNCTION.finditer(string):
                c_functions.add(match.group("function"))

        return natsort.natsorted(c_functions)

    dsp = Path(file).read_bytes().replace(b"\0", b"")
    dsp = lxml.etree.fromstring(dsp)

    channels = parse_channels(dsp.find("DISPLAY_INFO"))
    c_functions = parse_c_functions(dsp)

    functions = {}
    virtual_channels = []

    for i, ch in enumerate(
        parse_virtual_channels(dsp.find("VIRTUAL_CHANNEL")).values()
    ):
        virtual_channels.append(
            {
                "color": COLORS[i % len(COLORS)],
                "common_axis": False,
                "computed": True,
                "computation": {
                    "args": {"arg1": []},
                    "type": "python_function",
                    "channel_comment": ch["comment"],
                    "channel_name": ch["name"],
                    "channel_unit": "",
                    "function": f"f_{ch['name']}",
                    "triggering": "triggering_on_all",
                    "triggering_value": "all",
                },
                "flags": int(
                    Signal.Flags.computed | Signal.Flags.user_defined_conversion
                ),
                "enabled": True,
                "fmt": "{}",
                "individual_axis": False,
                "name": ch["parent"],
                "precision": 3,
                "ranges": [],
                "unit": "",
                "conversion": ch["vtab"],
                "user_defined_name": ch["name"],
                "comment": f"Datalyser virtual channel: {ch['comment']}",
                "origin_uuid": "000000000000",
                "type": "channel",
            }
        )

        functions[
            f"f_{ch['name']}"
        ] = f"def f_{ch['name']}(arg1=0, t=0):\n    return arg1"

    if virtual_channels:
        channels.append(
            {
                "name": "Datalyser Virtual Channels",
                "enabled": False,
                "type": "group",
                "channels": virtual_channels,
                "pattern": None,
                "origin_uuid": "000000000000",
                "ranges": [],
            }
        )

    info = {
        "selected_channels": [],
        "windows": [],
        "has_virtual_channels": bool(virtual_channels),
        "c_functions": c_functions,
        "functions": functions,
    }

    if flat:
        info = flatten_dsp(channels)
    else:

        plot = {
            "type": "Plot",
            "title": "Display channels",
            "maximized": True,
            "configuration": {
                "channels": channels,
                "locked": True,
            },
        }

        info["windows"].append(plot)

    return info


def flatten_dsp(channels):
    res = []

    for item in channels:
        if item["type"] == "group":
            res.extend(flatten_dsp(item["channels"]))
        else:
            res.append(item["name"])

    return res


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


class ProgressDialog(QtWidgets.QProgressDialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def keyPressEvent(self, event):
        if (
            event.key() == QtCore.Qt.Key_Escape
            and event.modifiers() == QtCore.Qt.NoModifier
        ):
            self.close()
        else:
            super().keyPressEvent(event)


def setup_progress(parent, title, message, icon_name):
    progress = ProgressDialog(message, "Cancel", 0, 100, parent)

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


def get_data_function(signals, fill_0_for_missing_computation_channels):
    def get_data(name, t=0, interpolated=False):
        if name in signals:
            samples = (
                signals[name]
                .interp(
                    [t],
                    integer_interpolation_mode=IntegerInterpolation.REPEAT_PREVIOUS_SAMPLE
                    if not interpolated
                    else IntegerInterpolation.LINEAR_INTERPOLATION,
                    float_interpolation_mode=FloatInterpolation.REPEAT_PREVIOUS_SAMPLE
                    if not interpolated
                    else FloatInterpolation.LINEAR_INTERPOLATION,
                )
                .samples
            )
            if len(samples):
                return samples[0]
            else:
                return 0

        elif fill_0_for_missing_computation_channels:
            return 0

        else:
            raise Exception(f"{name} channel was not found")

    return get_data


def compute_signal(
    description,
    measured_signals,
    all_timebase,
    functions,
):
    type_ = description["type"]

    try:
        if type_ == "python_function":

            func, trace = (
                None,
                f"{description['function']} not found in the user defined functions",
            )

            _globals = {
                "math": math,
                "np": np,
                "pd": pd,
            }

            for function_name, definition in functions.items():
                _func, _trace = generate_python_function(definition, _globals)

                if function_name == description["function"]:
                    func, trace = _func, _trace

            if func is None:
                raise Exception(trace)

            signals = []
            found_args = []

            for arg, alternative_names in description["args"].items():
                for name in alternative_names:
                    if name in measured_signals:
                        signals.append(measured_signals[name])
                        found_args.append(arg)
                        break

            names = found_args + ["t"]

            triggering = description.get("triggering", "triggering_on_all")
            if triggering == "triggering_on_all":
                timestamps = [sig.timestamps for sig in signals]

                if timestamps:
                    common_timebase = reduce(np.union1d, timestamps)
                else:
                    common_timebase = all_timebase
                signals = [
                    sig.interp(common_timebase).samples.tolist() for sig in signals
                ]

            elif triggering == "triggering_on_channel":
                triggering_channel = description["triggering_value"]

                if triggering_channel in measured_signals:
                    common_timebase = measured_signals[triggering_channel].timestamps
                else:
                    common_timebase = np.array([])
                signals = [
                    sig.interp(common_timebase).samples.tolist() for sig in signals
                ]
            else:

                step = float(description["triggering_value"])

                common_timebase = []
                for signal in signals:
                    if len(signal):
                        common_timebase.append(signal.timestamps[0])
                        common_timebase.append(signal.timestamps[-1])

                common_timebase = common_timebase or all_timebase

                if common_timebase:
                    common_timebase = np.unique(common_timebase)
                    start = common_timebase[0]
                    stop = common_timebase[-1]

                    common_timebase = np.arange(start, stop, step)

                else:
                    common_timebase = np.array([])

                signals = [
                    sig.interp(common_timebase).samples.tolist() for sig in signals
                ]

            signals.append(common_timebase)

            samples = [
                func(**{arg_name: arg_val for arg_name, arg_val in zip(names, values)})
                for values in zip(*signals)
            ]

            result = Signal(
                name="_",
                samples=samples,
                timestamps=common_timebase,
                flags=Signal.Flags.computed,
            )

    except:
        print(format_exc())
        result = Signal(
            name="_",
            samples=[],
            timestamps=[],
            flags=Signal.Flags.computed,
        )

    return result


def computation_to_python_function(description):
    type_ = description["type"]

    if type_ == "arithmetic":
        op = description["op"]

        args = []
        fargs = {}

        operand1 = description["operand1"]
        if isinstance(operand1, dict):
            fargs["arg1"] = []
            args.append("arg1=0")
            operand1 = "arg1"

        elif isinstance(operand1, str):
            try:
                operand1 = float(operand1)
                if operand1.is_integer():
                    operand1 = int(operand1)
            except:

                fargs["arg1"] = [operand1]
                args.append("arg1=0")
                operand1 = "arg1"

        operand2 = description["operand2"]
        if isinstance(operand2, dict):
            fargs["arg2"] = []
            args.append("arg2=0")
            operand2 = "arg2"
        elif isinstance(operand2, str):
            try:
                operand2 = float(operand2)
                if operand2.is_integer():
                    operand2 = int(operand2)
            except:

                fargs["arg2"] = [operand2]
                args.append("arg2=0")
                operand2 = "arg2"

        args.append("t=0")

        function_name = f"Arithmetic_{os.urandom(6).hex()}"
        args = ", ".join(args)
        body = f"return {operand1} {op} {operand2}"

        definition = f"def {function_name}({args}):\n    {body}"

        new_description = {
            "args": fargs,
            "channel_comment": description["channel_comment"],
            "channel_name": description["channel_name"],
            "channel_unit": description["channel_unit"],
            "definition": definition,
            "type": "python_function",
            "triggering": "triggering_on_all",
            "triggering_value": "all",
            "function": function_name,
        }

    elif type_ == "function":
        channel = description["channel"]

        args = []
        fargs = {}

        if isinstance(channel, dict):
            fargs["arg1"] = []
            operand = "arg1"
            args.append("arg1=0")

        elif isinstance(channel, str):
            fargs["arg1"] = [channel]
            operand = "arg1"
            args.append("arg1=0")

        args.append("t=0")

        np_args = ", ".join([operand] + [str(e) for e in description["args"]])
        np_function = description["name"]

        function_name = f"Numpy_{os.urandom(6).hex()}"
        args = ", ".join(args)
        body = f"return np.{np_function}( {np_args} )"

        definition = f"def {function_name}({args}):\n    {body}"

        new_description = {
            "args": fargs,
            "channel_comment": description["channel_comment"],
            "channel_name": description["channel_name"],
            "channel_unit": description["channel_unit"],
            "definition": definition,
            "type": "python_function",
            "triggering": "triggering_on_all",
            "triggering_value": "all",
            "function": function_name,
        }

    elif type_ == "expression":
        exp = description["expression"]

        args = []
        fargs = {}

        translation = {}

        for match in VARIABLE.finditer(exp):
            name = match.group("var")
            if name not in translation:
                arg = f"arg{len(translation)+1}"
                translation[name] = arg
                args.append(f"{arg}=0")
                fargs[arg] = [name.strip("}{")]

        args.append("t=0")

        for name, arg in translation.items():
            exp = exp.replace(name, arg)

        function_name = f"Expression_{os.urandom(6).hex()}"
        args = ", ".join(args)
        body = f"return {exp}"

        definition = f"def {function_name}({args}):\n    {body}"

        new_description = {
            "args": fargs,
            "channel_comment": description["channel_comment"],
            "channel_name": description["channel_name"],
            "channel_unit": description["channel_unit"],
            "definition": definition,
            "type": "python_function",
            "triggering": "triggering_on_all",
            "triggering_value": "all",
            "function": function_name,
        }

    else:
        if "args" not in description or "function" not in description:
            exp = description["definition"]

            args = []
            fargs = {}

            translation = {}

            for match in VARIABLE.finditer(exp):
                name = match.group("var")
                if name not in translation:
                    arg = f"arg{len(translation) + 1}"
                    translation[name] = arg
                    args.append(f"{arg}=0")
                    fargs[arg] = [name.strip("}{")]

            args.append("t=0")

            for name, arg in translation.items():
                exp = exp.replace(name, arg)

            function_name = description["channel_name"]
            args = ", ".join(args)
            body = indent(exp, "    ", lambda line: True)

            definition = f"def {function_name}({args}):\n    {body}"

            new_description = {
                "args": fargs,
                "channel_comment": description["channel_comment"],
                "channel_name": description["channel_name"],
                "channel_unit": description["channel_unit"],
                "definition": definition,
                "type": "python_function",
                "triggering": "triggering_on_all",
                "triggering_value": "all",
                "function": function_name,
            }
        else:
            new_description = description

    return new_description


def replace_computation_dependency(computation, old_name, new_name):
    new_computation = {}
    for key, val in computation:
        if isinstance(val, str) and old_name in val:
            new_computation[key] = val.replace(old_name, new_name)
        elif isinstance(val, dict):
            new_computation[key] = replace_computation_dependency(
                val, old_name, new_name
            )
        else:
            new_computation[key] = val

    return new_computation


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
                    range_info[color_name] = fn.mkColor(color)
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
        if isinstance(value, (float, int, np.number)):
            level_class = float
        else:
            level_class = str

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

            if isinstance(value1, level_class):
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

            if isinstance(value2, level_class):
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

    return new_background_color, new_font_color


def get_color_using_ranges(
    value,
    ranges,
    default_color,
    pen=False,
):
    new_color = default_color

    if value is None:
        return new_color

    if ranges:
        if isinstance(value, (float, int, np.number)):
            level_class = float
        else:
            level_class = str

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

            if isinstance(value1, level_class):
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

            if isinstance(value2, level_class):
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

                new_color = font_color
                break

    if pen:
        return fn.mkPen(new_color.name())
    else:
        return new_color


def timeit(func):
    def timed(*args, **kwargs):
        t1 = perf_counter()
        ret = func(*args, **kwargs)
        t2 = perf_counter()
        delta = t2 - t1
        if delta >= 1e-3:
            print(f"CALL {func.__qualname__}: {delta*1e3:.3f} ms")
        else:
            print(f"CALL {func.__qualname__}: {delta*1e6:.3f} us")
        return ret

    return timed


def value_as_bin(value, dtype):
    byte_string = np.array([value], dtype=dtype).tobytes()
    if dtype.byteorder != ">":
        byte_string = byte_string[::-1]

    nibles = []
    for byte in byte_string:
        nibles.append(f"{byte >> 4:04b}")
        nibles.append(f"{byte & 0xf:04b}")

    return ".".join(nibles)


def value_as_hex(value, dtype):
    byte_string = np.array([value], dtype=dtype).tobytes()
    if dtype.byteorder != ">":
        byte_string = byte_string[::-1]

    return f"0x{byte_string.hex().upper()}"


def value_as_str(value, format, dtype=None, precision=3):

    float_fmt = f"{{:.0{precision}f}}" if precision >= 0 else "{}"
    if isinstance(value, (float, np.floating)):
        kind = "f"

    elif isinstance(value, int):
        kind = "u"
        value = np.min_scalar_type(value).type(value)
        dtype = dtype or value.dtype

    elif isinstance(value, np.integer):
        kind = "u"
        dtype = value.dtype

    else:
        kind = "S"

    if kind in "ui":
        if format == "bin":
            string = value_as_bin(value, dtype)
        elif format == "hex":
            string = value_as_hex(value, dtype)
        elif format == "ascii":
            if 0 < value < 0x110000:
                string = chr(value)
            else:
                string = str(value)
        else:
            string = str(value)
    elif kind in "SUV":
        string = str(value)
    else:
        string = float_fmt.format(value)

    return string


def draw_color_icon(color):
    color = QtGui.QColor(color)
    pix = QtGui.QPixmap(64, 64)
    painter = QtGui.QPainter(pix)
    painter.setPen("black")
    painter.drawRect(QtCore.QRect(0, 0, 63, 63))
    painter.setPen(color)
    painter.setBrush(color)
    painter.drawRect(QtCore.QRect(1, 1, 62, 62))
    painter.end()
    return QtGui.QIcon(pix)


def generate_python_function(definition, in_globals=None):
    trace = None
    func = None

    definition = definition.replace("\t", "    ")

    _globals = in_globals or {}
    _globals.update(
        {
            "math": math,
            "np": np,
            "pd": pd,
        }
    )

    if not definition:
        trace = "The function definition must not be empty"
        return func, trace

    function_name = ""
    for match in FUNC_NAME.finditer(definition):
        function_name = match.group("name")

    if not function_name:
        trace = "The function name must not be empty"
        return func, trace

    try:
        exec(definition, _globals)
        func = _globals[function_name]
    except:
        trace = format_exc()
        func = None

    if func is not None:

        args = inspect.signature(func)
        if "t" not in args.parameters:
            trace = 'The last function argument must be "t=0"'
            func = None

        else:
            count = len(args.parameters)

            for i, (arg_name, arg) in enumerate(args.parameters.items()):
                if i == count - 1:
                    if arg_name != "t":
                        trace = 'The last function argument must be "t=0"'
                        func = None

                    elif arg.default != 0:
                        trace = 'The last function argument must be "t=0"'
                        func = None
                else:
                    if arg.default == inspect._empty:
                        trace = f'All the arguments must have default values. The argument "{arg_name}" has no default value.'
                        func = None
                        break

    return func, trace


if __name__ == "__main__":
    pass

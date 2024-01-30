import ctypes
from datetime import datetime
from functools import reduce
import inspect
from io import StringIO
import math
import os
import random
import re
import sys
from textwrap import indent
from threading import Thread
from time import sleep
import traceback
from traceback import format_exc
from typing import Dict, Union

import numpy as np
import pandas as pd
from pyqtgraph import functions as fn
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QThreadPool

from ..blocks.options import FloatInterpolation, IntegerInterpolation
from ..signal import Signal
from .dialogs.error_dialog import ErrorDialog
from .dialogs.messagebox import MessageBox

ERROR_ICON = None
RANGE_INDICATOR_ICON = None
NO_ERROR_ICON = None

COMPUTED_FUNCTION_ERROR_VALUE = float("nan")


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
COLORS_COUNT = len(COLORS)

GREEN = "#599e5e"
BLUE = "#61b2e2"

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
    print(f"{exc_type}: {exc_value}")

    ErrorDialog(message=errmsg, trace=msg, title="The following error was triggered").exec_()


def run_thread_with_progress(widget, target, kwargs, factor=100, offset=0, progress=None):
    termination_request = False

    thr = WorkerThread(target=target, kwargs=kwargs)

    thr.start()

    while widget.progress is None and thr.is_alive():
        sleep(0.02)

    while thr.is_alive():
        if progress and not progress.wasCanceled():
            if widget.progress is not None:
                if widget.progress != (0, 0):
                    progress.setValue(int(widget.progress[0] / widget.progress[1] * factor) + offset)
                else:
                    progress.setRange(0, 0)
        QtCore.QCoreApplication.processEvents()
        sleep(0.1)

    if progress:
        progress.setValue(factor + offset)

    if thr.error:
        widget.progress = None
        if progress:
            progress.cancel()
        raise Exception(thr.error)

    widget.progress = None

    if termination_request:
        return TERMINATED
    else:
        return thr.output


class WorkerSignals(QtCore.QObject):
    finished = QtCore.Signal()
    error = QtCore.Signal(object)
    result = QtCore.Signal(object)

    # will forward to the progress dialog
    setLabelText = QtCore.Signal(str)
    setWindowTitle = QtCore.Signal(str)
    setWindowIcon = QtCore.Signal(object)
    setMinimum = QtCore.Signal(int)
    setMaximum = QtCore.Signal(int)
    setValue = QtCore.Signal(int)


class Worker(QtCore.QRunnable):
    def __init__(self, function, *args, **kwargs):
        super().__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.stop = False
        self.TERMINATED = TERMINATED

        # Add the qrunner to the keyword arguments
        kwargs["progress"] = self

    @QtCore.Slot()
    def run(self):
        try:
            result = self.function(*self.args, **self.kwargs)
        except Exception:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()


class ProgressDialog(QtWidgets.QProgressDialog):
    qfinished = QtCore.Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threadpool = QThreadPool()
        self.worker = None
        self.error = None
        self.result = None
        self.thread_finished = True
        # Connect signal to "processEvents": Give the chance to "destroy" function to make his job
        self.qfinished.connect(lambda: QtCore.QCoreApplication.processEvents())

    def run_thread_with_progress(self, target, args, kwargs):
        self.show()
        self.result = None
        self.error = None
        self.thread_finished = False

        self.worker = Worker(target, *args, **kwargs)
        self.worker.signals.result.connect(self.receive_result)
        self.worker.signals.finished.connect(self.thread_complete)
        self.worker.signals.error.connect(self.receive_error)

        self.worker.signals.setLabelText.connect(self.setLabelText)
        self.worker.signals.setWindowIcon.connect(self.setWindowIcon)
        self.worker.signals.setWindowTitle.connect(self.setWindowTitle)
        self.worker.signals.setValue.connect(self.setValue)
        self.worker.signals.setMinimum.connect(self.setMinimum)
        self.worker.signals.setMaximum.connect(self.setMaximum)

        self.canceled.connect(self._canceled)

        self.threadpool.start(self.worker)

    def _canceled(self):
        self.close()

    def receive_result(self, result):
        self.result = result

    def receive_error(self, error):
        self.error = error

    def thread_complete(self):
        self.thread_finished = True
        super().close()
        self.qfinished.emit()

    def cancel(self):
        super().cancel()
        self.destroy()

    def close(self):
        while not self.thread_finished:
            self.worker.stop = True
            sleep(0.01)
            QtCore.QCoreApplication.processEvents()
        self.destroy()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_Escape and event.modifiers() == QtCore.Qt.KeyboardModifier.NoModifier:
            event.accept()
            self.close()
        else:
            super().keyPressEvent(event)


def setup_progress(parent, title="", message="", icon_name="", autoclose=False):
    progress = ProgressDialog(message, "Cancel", 0, 100, parent)

    progress.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
    progress.setCancelButton(None)
    progress.setAutoClose(autoclose)
    progress.setWindowTitle(title)
    icon = QtGui.QIcon()
    icon.addPixmap(QtGui.QPixmap(f":/{icon_name}.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
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
                    integer_interpolation_mode=(
                        IntegerInterpolation.REPEAT_PREVIOUS_SAMPLE
                        if not interpolated
                        else IntegerInterpolation.LINEAR_INTERPOLATION
                    ),
                    float_interpolation_mode=(
                        FloatInterpolation.REPEAT_PREVIOUS_SAMPLE
                        if not interpolated
                        else FloatInterpolation.LINEAR_INTERPOLATION
                    ),
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
    required_channels = {}
    for key, sig in measured_signals.items():
        signal = sig.physical(copy=False)
        if signal.samples.dtype.kind in "fui":
            required_channels[key] = signal
        else:
            required_channels[key] = sig

    measured_signals = required_channels

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
                    else:
                        found_numeric = False
                        for base in (10, 16, 2):
                            try:
                                value = int(name, base)
                                signals.append(value)
                                found_args.append(arg)
                                found_numeric = True
                                break
                            except:
                                continue

                        else:
                            try:
                                value = float(name)
                                signals.append(value)
                                found_args.append(arg)
                                found_numeric = True
                            except:
                                continue

                        if found_numeric:
                            break

            names = [*found_args, "t"]

            triggering = description.get("triggering", "triggering_on_all")
            if triggering == "triggering_on_all":
                timestamps = [sig.timestamps for sig in signals if not isinstance(sig, (int, float))]

                if timestamps:
                    common_timebase = reduce(np.union1d, timestamps)
                else:
                    common_timebase = all_timebase
                signals = [sig.interp(common_timebase) if not isinstance(sig, (int, float)) else sig for sig in signals]

            elif triggering == "triggering_on_channel":
                triggering_channel = description["triggering_value"]

                if triggering_channel in measured_signals:
                    common_timebase = measured_signals[triggering_channel].timestamps
                else:
                    common_timebase = np.array([])
                signals = [sig.interp(common_timebase) if not isinstance(sig, (int, float)) else sig for sig in signals]
            else:
                step = float(description["triggering_value"])

                common_timebase = []
                for signal in signals:
                    if isinstance(signal, (int, float)):
                        continue

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

                signals = [sig.interp(common_timebase) if not isinstance(sig, (int, float)) else sig for sig in signals]

            for i, (signal, arg_name) in enumerate(zip(signals, found_args)):
                if isinstance(signal, (int, float)):
                    value = signal
                    signals[i] = Signal(
                        name=arg_name, samples=np.full(len(common_timebase), value), timestamps=common_timebase
                    )

            if description.get("computation_mode", "sample_by_sample") == "sample_by_sample":
                signals = [sig.samples.tolist() for sig in signals]
                signals.append(common_timebase)

                samples = []
                for values in zip(*signals):
                    try:
                        current_sample = func(**dict(zip(names, values)))
                    except:
                        current_sample = COMPUTED_FUNCTION_ERROR_VALUE
                    samples.append(current_sample)

                result = Signal(
                    name="_",
                    samples=samples,
                    timestamps=common_timebase,
                    flags=Signal.Flags.computed,
                )

            else:
                signals = [sig.samples for sig in signals]

                signals.append(common_timebase)

                not_found = [arg_name for arg_name in description["args"] if arg_name not in names]

                not_found_signals = []

                args = inspect.signature(func)

                for i, (arg_name, arg) in enumerate(args.parameters.items()):
                    if arg_name in names:
                        continue
                    else:
                        not_found_signals.append(np.ones(len(common_timebase), dtype="u1") * arg.default)

                names.extend(not_found)
                signals.extend(not_found_signals)

                samples = func(**dict(zip(names, signals)))
                if len(samples) != len(common_timebase):
                    common_timebase = common_timebase[-len(samples) :]

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
            "computation_mode": "sample_by_sample",
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
            "computation_mode": "sample_by_sample",
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
            "channel_comment": description.get("channel_comment", ""),
            "channel_name": description.get("channel_name", ""),
            "channel_unit": description.get("channel_unit", ""),
            "computation_mode": "sample_by_sample",
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
                "computation_mode": description.get("computation_mode", "sample_by_sample"),
                "definition": definition,
                "type": "python_function",
                "triggering": "triggering_on_all",
                "triggering_value": "all",
                "function": function_name,
            }
        else:
            new_description = description
            new_description["computation_mode"] = description.get("computation_mode", "sample_by_sample")

    return new_description


def replace_computation_dependency(computation, old_name, new_name):
    new_computation = {}
    for key, val in computation:
        if isinstance(val, str) and old_name in val:
            new_computation[key] = val.replace(old_name, new_name)
        elif isinstance(val, dict):
            new_computation[key] = replace_computation_dependency(val, old_name, new_name)
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


def unique_ranges(ranges):
    def compare(r):
        v1, v2, op1, op2, c1, c2 = r
        if v1 is None:
            v1 = -float("inf")
        if v2 is None:
            v2 = -float("inf")

        return v1, v2, op1, op2, c1, c2

    if ranges:
        new_ranges = set()
        for range_info in ranges:
            if isinstance(range_info["background_color"], QtGui.QColor):
                new_ranges.add(
                    (
                        range_info["value1"],
                        range_info["value2"],
                        range_info["op1"],
                        range_info["op2"],
                        (0, range_info["background_color"].name()),
                        (0, range_info["font_color"].name()),
                    )
                )
            else:
                new_ranges.add(
                    (
                        range_info["value1"],
                        range_info["value2"],
                        range_info["op1"],
                        range_info["op2"],
                        (1, range_info["background_color"].color().name()),
                        (1, range_info["font_color"].color().name()),
                    )
                )

        ranges = []
        for value1, value2, op1, op2, bk_color, ft_color in sorted(new_ranges, key=compare):
            if bk_color[0] == 0:
                ranges.append(
                    {
                        "background_color": fn.mkColor(bk_color[1]),
                        "font_color": fn.mkColor(ft_color[1]),
                        "op1": op1,
                        "op2": op2,
                        "value1": value1,
                        "value2": value2,
                    }
                )
            else:
                ranges.append(
                    {
                        "background_color": fn.mkBrush(bk_color[1]),
                        "font_color": fn.mkBrush(ft_color[1]),
                        "op1": op1,
                        "op2": op2,
                        "value1": value1,
                        "value2": value2,
                    }
                )

    return ranges


def get_colors_using_ranges(value, ranges, default_background_color, default_font_color):
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


def value_as_bin(value, dtype):
    byte_string = np.array([value], dtype=dtype).tobytes()
    if dtype.byteorder != ">":
        byte_string = byte_string[::-1]

    nibles = []
    for byte in byte_string:
        nibles.extend((f"{byte >> 4:04b}", f"{byte & 0xf:04b}"))

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


def generate_python_function(definition: str, in_globals: Union[Dict, None] = None) -> tuple:
    trace = None
    func = None

    if not isinstance(definition, str):
        trace = "The function definition must be a string"
        return func, trace
    if in_globals and not isinstance(in_globals, dict):
        trace = "'in_globals' must be a dict"
        return func, trace

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


def check_generated_function(func, trace, function_source, silent, parent=None):
    if trace is not None:
        ErrorDialog(
            title="Function definition check",
            message="The syntax is not correct. The following error was found",
            trace=f"{trace}\n\nin the function\n\n{function_source}",
            parent=parent,
        ).exec()
        return False, None

    args = inspect.signature(func)
    kwargs = {}
    for i, (arg_name, arg) in enumerate(args.parameters.items()):
        kwargs[arg_name] = random.randint(1, 2**64)

    trace = ""

    # try with sample by sample call
    sample_by_sample = True
    try:
        func(**kwargs)
    except ZeroDivisionError:
        pass
    except:
        sample_by_sample = False
        trace = format_exc()

    kwargs = {}
    for i, (arg_name, arg) in enumerate(args.parameters.items()):
        kwargs[arg_name] = np.ones(10000, dtype="i1") * random.randint(1, 2**64)

    # try with complete signal call
    complete_signal = True
    try:
        func(**kwargs)
    except ZeroDivisionError:
        pass
    except:
        complete_signal = False
        if trace:
            trace += "\n\n" + format_exc()
        else:
            trace = format_exc()

    if not sample_by_sample and not complete_signal:
        ErrorDialog(
            title="Function definition check",
            message="The syntax is not correct. The following error was found",
            trace=f"{trace}\n\nin the function\n\n{function_source}",
            parent=parent,
        ).exec()

        return False, None

    elif not sample_by_sample:
        if not silent:
            MessageBox.information(
                parent,
                "Function definition check",
                "The function definition appears to be correct only for complete signal mode.",
            )

        return True, func

    elif not complete_signal:
        if not silent:
            MessageBox.information(
                parent,
                "Function definition check",
                "The function definition appears to be correct only for sample by sample mode.",
            )

        return True, func
    else:
        if not silent:
            MessageBox.information(
                parent,
                "Function definition check",
                "The function definition appears to be correct for both sample by sample and complete signal modes.",
            )

        return True, func


def set_app_user_model_id(app_user_model_id: str) -> None:
    if sys.platform == "win32":
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_user_model_id)


if __name__ == "__main__":
    pass

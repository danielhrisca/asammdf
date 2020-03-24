# -*- coding: utf-8 -*-
import traceback

from datetime import datetime
from io import StringIO
from time import sleep
from struct import unpack
from threading import Thread
from pathlib import Path
import lxml
import natsort
import numpy as np

from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5 import QtCore

from ..mdf import MDF, MDF2, MDF3, MDF4
from ..signal import Signal
from .dialogs.error_dialog import ErrorDialog


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
    names = []
    if data.hasFormat("application/octet-stream-asammdf"):
        data = bytes(data.data("application/octet-stream-asammdf"))
        size = len(data)
        pos = 0
        while pos < size:
            group_index, channel_index, name_length = unpack(
                "<3q", data[pos : pos + 24]
            )
            pos += 24
            name = data[pos : pos + name_length].decode("utf-8")
            pos += name_length
            names.append((name, group_index, channel_index))
    return names


def load_dsp(file):
    def parse_dsp(display, level=1):
        channels = set()
        groups = []
        all_channels = set()

        for item in display.findall("CHANNEL"):
            channels.add(item.get("name"))
            all_channels = all_channels | channels

        for item in display.findall(f"GROUP{level}"):
            group_channels, subgroups, subgroup_all_channels = parse_dsp(
                item, level + 1
            )
            all_channels = all_channels | subgroup_all_channels
            groups.append(
                {
                    "name": item.get("data"),
                    "dsp_channels": natsort.natsorted(group_channels),
                    "dsp_groups": subgroups,
                }
            )

        return channels, groups, all_channels

    dsp = Path(file).read_bytes().replace(b"\0", b"")
    dsp = lxml.etree.fromstring(dsp)

    channels, groups, all_channels = parse_dsp(dsp.find("DISPLAY_INFO"))

    info = {}
    all_channels = natsort.natsorted(all_channels)
    info["selected_channels"] = all_channels

    info["windows"] = windows = []

    numeric = {
        "type": "Numeric",
        "title": "Numeric",
        "configuration": {
            "channels": all_channels,
            "format": "phys",
        }
    }

    windows.append(numeric)

    plot = {
        "type": "Plot",
        "title": "Plot",
        "configuration": {
            "channels": [
                {
                    "color": COLORS[i%len(COLORS)],
                    "common_axis": False,
                    "computed": False,
                    "enabled": True,
                    "fmt": "{}",
                    "individual_axis": False,
                    "name": name,
                    "precision": 3,
                    "ranges": [],
                    "unit": "",
                }
                for i, name in enumerate(all_channels)
            ]
        }
    }

    windows.append(plot)

    return info


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
                progress.setValue(
                    int(widget.progress[0] / widget.progress[1] * factor) + offset
                )
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


def get_required_signals(channel):
    names = []
    if "computed" in channel:
        if channel["computed"]:
            computation = channel["computation"]
            if computation["type"] == "arithmetic":
                for op in (
                    computation["operand1"],
                    computation["operand2"],
                ):
                    if isinstance(op, str):
                        names.append(op)
                    elif isinstance(op, (int, float)):
                        pass
                    else:
                        names.extend(get_required_signals(op))
            else:
                op = computation["channel"]
                if isinstance(op, str):
                    names.append(op)
                else:
                    names.extend(get_required_signals(op))
        else:
            names.append(channel["name"])
    else:
        if channel["type"] == "arithmetic":
            for op in (channel["operand1"], channel["operand2"]):
                if isinstance(op, str):
                    names.append(op)
                elif isinstance(op, (int, float)):
                    pass
                else:
                    names.extend(get_required_signals(op))
        else:
            op = channel["channel"]
            if isinstance(op, str):
                names.append(op)
            else:
                names.extend(get_required_signals(op))

    return names


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

    else:
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

        result = Signal(samples=samples, timestamps=timestamps, name="_",)

    return result
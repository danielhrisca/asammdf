"""
module to generate benchmark graphs from textul result file
"""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np


def generate_graphs(result: str, topic: str, aspect: Literal["ram", "time"], for_doc: bool = False) -> None:
    """genrate graphs from result file

    Parameters
    ----------
    result : str
        path to result file
    topic : str
        benchmark topic; for example "Open file" or "Save file"
    aspect : str
        performance indiitemsor; can be "ram" (RAM memory usage) or "time" (elapsed time)
    for_doc : bool
        wether the source code is used inside the documentation

    """
    with open(result) as f:
        lines = f.readlines()

    platform = "x86" if "32 bit" in lines[2] else "x64"

    idx = [i for i, line in enumerate(lines) if line.startswith("==")]

    table_spans = {
        "open": [idx[1] + 1, idx[2]],
        "save": [idx[4] + 1, idx[5]],
        "get": [idx[7] + 1, idx[8]],
        "convert": [idx[10] + 1, idx[11]],
        "merge": [idx[13] + 1, idx[14]],
    }

    start, stop = table_spans[topic.lower()]

    items = [l[:50].strip(" \t\r\n\0*") for l in lines[start:stop]]
    time = np.array([int(l[50:61].strip(" \t\r\n\0*")) for l in lines[start:stop]])
    ram = np.array([int(l[61:].strip(" \t\r\n\0*")) for l in lines[start:stop]])

    if aspect == "ram":
        array = ram
    else:
        array = time

    y_pos = list(range(len(items)))

    fig, ax = plt.subplots()
    fig.set_size_inches(15, 3.8 / 12 * len(items) + 1.2)

    asam_pos = [i for i, c in enumerate(items) if c.startswith("asam")]
    mdfreader_pos = [i for i, c in enumerate(items) if c.startswith("mdfreader")]

    ax.barh(asam_pos, array[asam_pos], color="green", ecolor="green")
    ax.barh(mdfreader_pos, array[mdfreader_pos], color="blue", ecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(items)
    ax.invert_yaxis()
    ax.set_xlabel("Time [ms]" if aspect == "time" else "RAM [MB]")
    if topic == "Get":
        ax.set_title("Get all channels (36424 calls) - {}".format("time" if aspect == "time" else "ram usage"))
    else:
        ax.set_title("{} test file - {}".format(topic, "time" if aspect == "time" else "ram usage"))
    ax.xaxis.grid()

    fig.subplots_adjust(
        bottom=0.72 / fig.get_figheight(),
        top=1 - 0.48 / fig.get_figheight(),
        left=0.4,
        right=0.9,
    )

    if aspect == "time":
        if topic == "Get":
            name = f"{platform}_get_all_channels.png"
        else:
            name = f"{platform}_{topic.lower()}.png"
    else:
        if topic == "Get":
            name = f"{platform}_get_all_channels_ram_usage.png"
        else:
            name = f"{platform}_{topic.lower()}_ram_usage.png"

    if for_doc:
        plt.show()
    else:
        plt.savefig(name, dpi=300)


files = (
    r"e:\02__PythonWorkspace\asammdf\benchmarks\results\x64_asammdf_2.7.0_mdfreader_0.2.7.txt",
    r"e:\02__PythonWorkspace\asammdf\benchmarks\results\x86_asammdf_2.7.0_mdfreader_0.2.7.txt",
)

for file in files:
    for topic in ("Open", "Save", "Get", "Convert", "Merge"):
        for aspect in ("time", "ram"):
            generate_graphs(file, topic, aspect)

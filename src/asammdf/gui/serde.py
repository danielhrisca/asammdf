from collections.abc import Collection
from copy import deepcopy
import json
from pathlib import Path
import re
from traceback import format_exc
import typing
from typing import Final, Literal, Union

import lxml.etree
from pyqtgraph import functions as fn
from PySide6 import QtCore, QtGui
from typing_extensions import NotRequired, TypedDict

from asammdf.blocks import v4_constants as v4c
from asammdf.blocks.utils import SignalFlags

C_FUNCTION: Final = re.compile(r"\s+(?P<function>\S+)\s*\(\s*struct\s+DATA\s+\*data\s*\)")
COLOR_MAPS: Final = {
    "Accent": ["#7fc97f", "#beaed4", "#fdc086", "#ffff99", "#386cb0", "#f0027f", "#bf5b16", "#666666"],
    "Dark2": ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666"],
    "Paired": [
        "#a6cee3",
        "#1f78b4",
        "#b2df8a",
        "#33a02c",
        "#fb9a99",
        "#e31a1c",
        "#fdbf6f",
        "#ff7f00",
        "#cab2d6",
        "#6a3d9a",
        "#ffff99",
        "#b15928",
    ],
    "Pastel1": ["#fbb4ae", "#b3cde3", "#ccebc5", "#decbe4", "#fed9a6", "#ffffcc", "#e5d8bd", "#fddaec", "#f2f2f2"],
    "Pastel2": ["#b3e2cd", "#fdcdac", "#cbd5e8", "#f4cae4", "#e6f5c9", "#fff2ae", "#f1e2cc", "#cccccc"],
    "Set1": ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628", "#f781bf", "#999999"],
    "Set2": ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3"],
    "Set3": [
        "#8dd3c7",
        "#ffffb3",
        "#bebada",
        "#fb8072",
        "#80b1d3",
        "#fdb462",
        "#b3de69",
        "#fccde5",
        "#d9d9d9",
        "#bc80bd",
        "#ccebc5",
        "#ffed6f",
    ],
    "tab10": [
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
    ],
    "tab20": [
        "#1f77b4",
        "#aec7e8",
        "#ff7f0e",
        "#ffbb78",
        "#2ca02c",
        "#98df8a",
        "#d62728",
        "#ff9896",
        "#9467bd",
        "#c5b0d5",
        "#8c564b",
        "#c49c94",
        "#e377c2",
        "#f7b6d2",
        "#7f7f7f",
        "#c7c7c7",
        "#bcbd22",
        "#dbdb8d",
        "#17becf",
        "#9edae5",
    ],
    "tab20b": [
        "#393b79",
        "#5254a3",
        "#6b6ecf",
        "#9c9ede",
        "#637939",
        "#8ca252",
        "#b5cf6b",
        "#cedb9c",
        "#8c6d31",
        "#bd9e39",
        "#e7ba52",
        "#e7cb94",
        "#843c39",
        "#ad494a",
        "#d6616b",
        "#e7969c",
        "#7b4173",
        "#a55194",
        "#ce6dbd",
        "#de9ed6",
    ],
    "tab20c": [
        "#3182bd",
        "#6baed6",
        "#9ecae1",
        "#c6dbef",
        "#e6550d",
        "#fd8d3c",
        "#fdae6b",
        "#fdd0a2",
        "#31a354",
        "#74c476",
        "#a1d99b",
        "#c7e9c0",
        "#756bb1",
        "#9e9ac8",
        "#bcbddc",
        "#dadaeb",
        "#636363",
        "#969696",
        "#bdbdbd",
        "#d9d9d9",
    ],
}
COLORS: Final = COLOR_MAPS["tab10"]
COLORS_COUNT: Final = len(COLORS)
COMPARISON_NAME: Final = re.compile(r"(\s*\d+:)?(?P<name>.+)")


class _ChannelBaseDict(TypedDict):
    color: str
    comment: str | None
    common_axis: bool
    enabled: bool
    flags: int
    fmt: str
    individual_axis: bool
    name: str
    origin_uuid: str
    precision: int
    ranges: list[dict[str, object]]
    type: Literal["channel"]
    unit: str


class _ChannelNotComputedDict(_ChannelBaseDict):
    computed: Literal[False]
    conversion: NotRequired[dict[str, object]]
    mode: Literal["phys"]
    y_range: list[float]


class _ChannelComputedDict(_ChannelBaseDict):
    computed: Literal[True]
    computation: dict[str, object]
    conversion: object
    user_defined_name: str | None


_ChannelDict = _ChannelComputedDict | _ChannelNotComputedDict


class _ChannelGroupDict(TypedDict):
    channels: list[Union["_ChannelGroupDict", _ChannelDict]]
    enabled: bool
    name: str | None
    origin_uuid: str
    pattern: dict[str, object] | None
    ranges: list[dict[str, object]]
    type: Literal["group"]


def extract_mime_names(data: QtCore.QMimeData, disable_new_channels: bool | None = None) -> list[str]:
    def fix_comparison_name(
        data: list[_ChannelGroupDict | _ChannelDict], disable_new_channels: bool | None = None
    ) -> None:
        for item in data:
            if item["type"] == "channel":
                if disable_new_channels is not None:
                    item["enabled"] = not disable_new_channels

                if (
                    item["group_index"],  # type: ignore[typeddict-item]
                    item["channel_index"],  # type: ignore[typeddict-item]
                ) != (-1, -1):
                    match = COMPARISON_NAME.match(item["name"])
                    if match is None:
                        raise RuntimeError(f"cannot parse '{item['name']}'")
                    name = match.group("name").strip()
                    item["name"] = name
            else:
                if disable_new_channels is not None:
                    item["enabled"] = not disable_new_channels
                fix_comparison_name(item["channels"], disable_new_channels=disable_new_channels)

    names: list[str] = []
    if data.hasFormat("application/octet-stream-asammdf"):
        data_data = data.data("application/octet-stream-asammdf").data()
        data_bytes = data_data.tobytes() if isinstance(data_data, memoryview) else data_data
        text = data_bytes.decode("utf-8")
        obj = json.loads(text, cls=ExtendedJsonDecoder)
        fix_comparison_name(obj, disable_new_channels=disable_new_channels)
        names = obj

    return names


def set_mime_enable(mime: list[_ChannelGroupDict | _ChannelDict], enable: bool) -> None:
    for item in mime:
        if item["type"] == "channel":
            item["enabled"] = enable
        else:
            set_mime_enable(item["channels"], enable)


def flatten_dsp(channels: list[_ChannelGroupDict | _ChannelDict]) -> list[str]:
    res: list[str] = []

    for item in channels:
        if item["type"] == "group":
            res.extend(flatten_dsp(item["channels"]))
        else:
            res.append(item["name"])

    return res


def load_dsp(
    file: Path, background: str = "#000000", flat: bool = False, colors_as_string: bool = False
) -> dict[str, object] | list[str]:
    if not colors_as_string and isinstance(background, str):
        background = fn.mkColor(background)

    def parse_conversions(display: lxml.etree._Element | None) -> dict[str | None, dict[str, object]]:
        conversions: dict[str | None, dict[str, object]] = {}

        if display is None:
            return conversions

        for item in display.findall("COMPU_METHOD"):
            try:
                name = item.get("name")
                conv: dict[str, object] = {
                    "name": name,
                    "comment": item.get("description"),
                    "unit": item.get("unit"),
                }

                conversion_type = int(item.attrib["cnv_type"])
                match conversion_type:
                    case 0:
                        conv["conversion_type"] = v4c.CONVERSION_TYPE_LIN

                        coeffs = item.find("COEFFS_LINIAR")

                        if coeffs is None:
                            raise RuntimeError("cannot find 'COEFFS_LINIAR' element")

                        conv["a"] = float(coeffs.attrib["P1"])
                        conv["b"] = float(coeffs.attrib["P2"])

                    case 9:
                        conv["conversion_type"] = v4c.CONVERSION_TYPE_RAT

                        coeffs = item.find("COEFFS")

                        if coeffs is None:
                            raise RuntimeError("cannot find 'COEFFS' element")

                        for i in range(1, 7):
                            conv[f"P{i}"] = float(coeffs.attrib[f"P{i}"])

                    case 11:
                        conv["conversion_type"] = v4c.CONVERSION_TYPE_TABX
                        vtab = item.find("COMPU_VTAB")

                        if vtab is not None:
                            for i, item in enumerate(vtab.findall("tab")):
                                conv[f"val_{i}"] = float(item.attrib["min"])
                                text = item.get("text")
                                if isinstance(text, bytes):
                                    text = text.decode("utf-8", errors="replace")
                                conv[f"text_{i}"] = text

                    case 12:
                        conv["conversion_type"] = v4c.CONVERSION_TYPE_RTABX
                        vtab = item.find("COMPU_VTAB_RANGE")

                        if vtab is not None:
                            text = vtab.get("default")
                            if isinstance(text, bytes):
                                text = text.decode("utf-8", errors="replace")
                            conv["default_addr"] = vtab.get("default")
                            for i, item in enumerate(vtab.findall("tab_range")):
                                conv[f"upper_{i}"] = float(item.attrib["max"])
                                conv[f"lower_{i}"] = float(item.attrib["min"])
                                text = item.get("text")
                                if isinstance(text, bytes):
                                    text = text.decode("utf-8", errors="replace")
                                conv[f"text_{i}"] = text
                    case _:
                        continue

                conversions[name] = conv

            except:
                print(format_exc())
                continue

        return conversions

    def parse_channels(
        display: lxml.etree._Element, conversions: dict[str | None, dict[str, object]]
    ) -> list[_ChannelGroupDict | _ChannelDict]:
        channels: list[_ChannelGroupDict | _ChannelDict] = []
        for elem in display.iterchildren():
            if elem.tag == "CHANNEL":
                channel_name = elem.attrib["name"] or "unnamed"

                comment_elem = elem.find("COMMENT")
                if comment_elem is not None:
                    comment = elem.get("text")
                else:
                    comment = ""

                color_ = int(elem.attrib["color"])
                c = 0
                for i in range(3):
                    c = c << 8
                    c += color_ & 0xFF
                    color_ = color_ >> 8

                ch_color = c

                gain = abs(float(elem.attrib["gain"]))
                offset = float(elem.attrib["offset"]) / 100

                multi_color = elem.find("MULTI_COLOR")

                ranges: list[dict[str, object]] = []

                if multi_color is not None:
                    for color in multi_color.findall("color"):
                        some_elem = color.find("min")
                        if some_elem is None:
                            raise RuntimeError("cannot find element 'min'")
                        min_ = float(some_elem.attrib["data"])
                        some_elem = color.find("max")
                        if some_elem is None:
                            raise RuntimeError("cannot find element 'max'")
                        max_ = float(some_elem.attrib["data"])
                        some_elem = color.find("color")
                        if some_elem is None:
                            raise RuntimeError("cannot find element 'color'")
                        color_ = int(some_elem.attrib["data"])
                        c = 0
                        for i in range(3):
                            c = c << 8
                            c += color_ & 0xFF
                            color_ = color_ >> 8
                        font_color = f"#{c:06X}" if colors_as_string else fn.mkColor(f"#{c:06X}")
                        ranges.append(
                            {
                                "background_color": background,
                                "font_color": font_color,
                                "op1": "<=",
                                "op2": "<=",
                                "value1": min_,
                                "value2": max_,
                            }
                        )

                chan: _ChannelNotComputedDict = {
                    "color": f"#{ch_color:06X}",
                    "common_axis": False,
                    "computed": False,
                    "flags": 0,
                    "comment": comment,
                    "enabled": elem.get("on") == "1" and elem.get("trc_fmt") != "2",
                    "fmt": "{}",
                    "individual_axis": False,
                    "name": channel_name,
                    "mode": "phys",
                    "precision": 3,
                    "ranges": ranges,
                    "unit": "",
                    "type": "channel",
                    "y_range": sorted(
                        [
                            -gain * offset,
                            -gain * offset + 19 * gain,
                        ]
                    ),
                    "origin_uuid": "000000000000",
                }

                conv_name = elem.get("cnv_name")
                if conv_name in conversions:
                    chan["conversion"] = deepcopy(conversions[conv_name])

                channels.append(chan)

            elif str(elem.tag).startswith("GROUP"):
                channels.append(
                    {
                        "name": elem.get("data") or "unnamed",
                        "enabled": elem.get("on") == "1",
                        "type": "group",
                        "channels": parse_channels(elem, conversions=conversions),
                        "pattern": None,
                        "origin_uuid": "000000000000",
                        "ranges": [],
                    }
                )

            elif elem.tag == "CHANNEL_PATTERN":
                try:
                    filter_type = elem.get("filter_type")
                    filter_value: float
                    if filter_type in ("None", None):
                        filter_type = "Unspecified"
                        filter_value = 0
                        raw = False
                    else:
                        filter_value = float(elem.attrib["filter_value"])
                        raw = bool(int(elem.attrib["filter_use_raw"]))

                    info: dict[str, object] = {
                        "pattern": elem.get("name_pattern"),
                        "name": elem.get("name_pattern") or "unnamed",
                        "match_type": "Wildcard",
                        "filter_type": filter_type,
                        "filter_value": filter_value,
                        "raw": raw,
                    }

                    multi_color = elem.find("MULTI_COLOR")

                    ranges = []

                    if multi_color is not None:
                        for color in multi_color.findall("color"):
                            some_elem = color.find("min")
                            if some_elem is None:
                                raise RuntimeError("cannot find element 'min'")
                            min_ = float(some_elem.attrib["data"])
                            some_elem = color.find("max")
                            if some_elem is None:
                                raise RuntimeError("cannot find element 'max'")
                            max_ = float(some_elem.attrib["data"])
                            some_elem = color.find("color")
                            if some_elem is None:
                                raise RuntimeError("cannot find element 'color'")
                            color_ = int(some_elem.attrib["data"])
                            c = 0
                            for i in range(3):
                                c = c << 8
                                c += color_ & 0xFF
                                color_ = color_ >> 8
                            font_color = f"#{c:06X}" if colors_as_string else fn.mkColor(f"#{c:06X}")
                            ranges.append(
                                {
                                    "background_color": background,
                                    "font_color": font_color,
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
                            "name": typing.cast(str, info["pattern"]),
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

    def parse_virtual_channels(display: lxml.etree._Element | None) -> dict[str | None, dict[str, object]]:
        channels: dict[str | None, dict[str, object]] = {}

        if display is None:
            return channels

        for item in display.findall("V_CHAN"):
            try:
                virtual_channel: dict[str, object] = {}

                parent = item.find("VIR_TIME_CHAN")
                vtab = item.find("COMPU_VTAB")
                if parent is None or vtab is None:
                    continue

                name = item.get("name")

                virtual_channel["name"] = name
                virtual_channel["parent"] = parent.attrib["data"]
                elem = item.find("description")
                if elem is None:
                    raise RuntimeError("cannot find element 'description'")
                virtual_channel["comment"] = elem.attrib["data"]

                conv: dict[str, object] = {}
                for i, item in enumerate(vtab.findall("tab")):
                    conv[f"val_{i}"] = float(item.attrib["min"])
                    text = item.get("text")
                    if isinstance(text, bytes):
                        text = text.decode("utf-8", errors="replace")
                    conv[f"text_{i}"] = text

                virtual_channel["vtab"] = conv

                channels[name] = virtual_channel
            except:
                continue

        return channels

    def parse_c_functions(display: lxml.etree._Element | None) -> Collection[str]:
        c_functions: set[str] = set()

        if display is None:
            return c_functions

        for item in display.findall("CALC_FUNC"):
            string = item.text

            if string is None:
                raise RuntimeError("element text is None")

            for match in C_FUNCTION.finditer(string):
                c_functions.add(match.group("function"))

        return sorted(c_functions)

    dsp = lxml.etree.fromstring(Path(file).read_bytes().replace(b"\0", b""), parser=lxml.etree.XMLParser(recover=True))

    conversions = parse_conversions(dsp.find("COMPU_METHODS"))

    elem = dsp.find("DISPLAY_INFO")

    if elem is None:
        raise RuntimeError("cannot find element 'DISPLAY_INFO'")

    channels = parse_channels(elem, conversions)
    c_functions = parse_c_functions(dsp)

    functions: dict[str, object] = {}
    virtual_channels: list[_ChannelGroupDict | _ChannelDict] = []

    for i, ch in enumerate(parse_virtual_channels(dsp.find("VIRTUAL_CHANNEL")).values()):
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
                "flags": int(SignalFlags.computed | SignalFlags.user_defined_conversion),
                "enabled": True,
                "fmt": "{}",
                "individual_axis": False,
                "name": typing.cast(str, ch["parent"]),
                "precision": 3,
                "ranges": [],
                "unit": "",
                "conversion": ch["vtab"],
                "user_defined_name": typing.cast(str | None, ch["name"]),
                "comment": f"Datalyser virtual channel: {ch['comment']}",
                "origin_uuid": "000000000000",
                "type": "channel",
            }
        )

        functions[f"f_{ch['name']}"] = f"def f_{ch['name']}(arg1=0, t=0):\n    return arg1"

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

    windows: list[dict[str, object]] = []
    info: dict[str, object] | list[str] = {
        "selected_channels": [],
        "windows": windows,
        "has_virtual_channels": bool(virtual_channels),
        "c_functions": c_functions,
        "functions": functions,
    }

    if flat:
        info = flatten_dsp(channels)
    else:
        plot: dict[str, object] = {
            "type": "Plot",
            "title": "Display channels",
            "maximized": True,
            "minimized": False,
            "configuration": {
                "channels": channels,
                "locked": True,
                "pattern": {},
            },
        }

        windows.append(plot)

    return info


class ExtendedJsonDecoder(json.JSONDecoder):
    def __init__(self, **kwargs):
        kwargs["object_hook"] = self.object_hook
        super().__init__(**kwargs)

    def object_hook(self, obj):
        for key in ("color", "background_color", "font_color"):
            if key in obj:
                obj[key] = fn.mkColor(obj[key])
        return obj


class ExtendedJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, QtGui.QBrush):
            return obj.color().name()
        elif isinstance(obj, QtGui.QColor):
            return obj.name()
        else:
            return super().default(obj)


def load_lab(file: Path) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    with open(file) as lab:
        for line in lab:
            line = line.strip()
            if not line:
                continue

            if line.startswith("[") and line.endswith("]"):
                section_name = line.strip("[]")
                s: list[str] = []
                sections[section_name] = s

            else:
                if "s" in locals():
                    s.append(line)

    return {name: channels for name, channels in sections.items() if channels if name != "SETTINGS"}


def load_channel_names_from_file(file_name: str, lab_section: str = "") -> list[str]:
    file_path = Path(file_name)
    channels: Collection[str]
    extension = file_path.suffix.lower()
    match extension:
        case ".dsp":
            channels = load_dsp(file_path, flat=True)

        case ".dspf":
            with open(file_path) as infile:
                info = json.load(infile, cls=ExtendedJsonDecoder)

            channels = []
            for window in info["windows"]:
                if window["type"] == "Plot":
                    channels.extend(flatten_dsp(window["configuration"]["channels"]))
                elif window["type"] == "Numeric":
                    channels.extend([item["name"] for item in window["configuration"]["channels"]])
                elif window["type"] == "Tabular":
                    channels.extend(window["configuration"]["channels"])

        case ".lab":
            info = load_lab(file_path)
            if info:
                if len(info) > 1 and lab_section:
                    channels = info[lab_section]
                else:
                    channels = list(info.values())[0]

                channels = [name.split(";")[0] for name in channels]

        case ".cfg":
            with open(file_path) as infile:
                info = json.load(infile, cls=ExtendedJsonDecoder)
            channels = info.get("selected_channels", [])
        case ".txt":
            try:
                with open(file_path) as infile:
                    info = json.load(infile, cls=ExtendedJsonDecoder)
                channels = info.get("selected_channels", [])
            except:
                with open(file_path) as infile:
                    channels = [line.strip() for line in infile.readlines()]
                    channels = [name for name in channels if name]

    return sorted(set(channels))

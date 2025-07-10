"""asammdf utility functions for channel conversions"""

from copy import deepcopy
import typing
from typing import Literal

from typing_extensions import overload

from . import v2_v3_blocks as v3b
from . import v2_v3_constants as v3c
from . import v4_blocks as v4b
from . import v4_constants as v4c
from .types import ChannelConversionType


@overload
def conversion_transfer(
    conversion: ChannelConversionType | None, version: Literal[2, 3] = ..., copy: bool = ...
) -> v3b.ChannelConversion: ...


@overload
def conversion_transfer(
    conversion: ChannelConversionType, version: Literal[4], copy: bool = ...
) -> v4b.ChannelConversion: ...


@overload
def conversion_transfer(conversion: None, version: Literal[4], copy: bool = ...) -> None: ...


@overload
def conversion_transfer(
    conversion: ChannelConversionType | None, version: Literal[2, 3, 4] = ..., copy: bool = ...
) -> ChannelConversionType | None: ...


def conversion_transfer(
    conversion: ChannelConversionType | None, version: Literal[2, 3, 4] = 3, copy: bool = False
) -> ChannelConversionType | None:
    """Convert between MDF4 and MDF3 channel conversions.

    Parameters
    ----------
    conversion : block
        Channel conversion.
    version : int, default 3
        Target MDF version.
    copy : bool, default False
        Return a copy if the input conversion version is the same as the
        required version.

    Returns
    -------
    conversion : block
        Channel conversion for specified version.
    """

    if version <= 3:
        if conversion is None:
            conversion = v3b.ChannelConversion(conversion_type=v3c.CONVERSION_TYPE_NONE)
        else:
            conversion_type = conversion.conversion_type
            if isinstance(conversion, v3b.ChannelConversion):
                if copy:
                    conversion = deepcopy(conversion)
            else:
                unit = conversion.unit.strip(" \r\n\t\0").encode("latin-1")

                match conversion_type:
                    case v4c.CONVERSION_TYPE_NON:
                        conversion = v3b.ChannelConversion(unit=unit, conversion_type=v3c.CONVERSION_TYPE_NONE)

                    case v4c.CONVERSION_TYPE_LIN:
                        conversion = v3b.ChannelConversion(
                            unit=unit,
                            conversion_type=v3c.CONVERSION_TYPE_LINEAR,
                            a=conversion.a,
                            b=conversion.b,
                        )

                    case v4c.CONVERSION_TYPE_RAT:
                        conversion = v3b.ChannelConversion(
                            unit=unit,
                            conversion_type=v3c.CONVERSION_TYPE_RAT,
                            P1=conversion.P1,
                            P2=conversion.P2,
                            P3=conversion.P3,
                            P4=conversion.P4,
                            P5=conversion.P5,
                            P6=conversion.P6,
                        )

                    case v4c.CONVERSION_TYPE_TAB:
                        v3_kwargs = v3b.ChannelConversionKwargs(unit=unit, conversion_type=v3c.CONVERSION_TYPE_TAB)
                        v3_kwargs["ref_param_nr"] = conversion.val_param_nr // 2
                        for i in range(conversion.val_param_nr // 2):
                            v3_kwargs[f"raw_{i}"] = conversion[f"raw_{i}"]  # type: ignore[literal-required]
                            v3_kwargs[f"phys_{i}"] = conversion[f"phys_{i}"]  # type: ignore[literal-required]

                        conversion = v3b.ChannelConversion(**v3_kwargs)

                    case v4c.CONVERSION_TYPE_TABI:
                        v3_kwargs = v3b.ChannelConversionKwargs(unit=unit, conversion_type=v3c.CONVERSION_TYPE_TABI)
                        v3_kwargs["ref_param_nr"] = conversion.val_param_nr // 2
                        for i in range(conversion.val_param_nr // 2):
                            v3_kwargs[f"raw_{i}"] = conversion[f"raw_{i}"]  # type: ignore[literal-required]
                            v3_kwargs[f"phys_{i}"] = conversion[f"phys_{i}"]  # type: ignore[literal-required]

                        conversion = v3b.ChannelConversion(**v3_kwargs)

                    case v4c.CONVERSION_TYPE_ALG:
                        formula = conversion.formula.replace("X", "X1")
                        conversion = v3b.ChannelConversion(
                            formula=formula,
                            unit=unit,
                            conversion_type=v3c.CONVERSION_TYPE_FORMULA,
                        )

                    case v4c.CONVERSION_TYPE_RTAB:
                        nr = (conversion.val_param_nr - 1) // 3
                        v3_kwargs = {
                            "unit": unit,
                            "ref_param_nr": nr,
                            "conversion_type": v3c.CONVERSION_TYPE_TABI,
                        }

                        for i in range(nr):
                            l_ = conversion[f"lower_{i}"]
                            u_ = typing.cast(float, conversion[f"upper_{i}"])
                            p_ = conversion[f"phys_{i}"]
                            v3_kwargs[f"raw_{i}"] = l_  # type: ignore[literal-required]
                            v3_kwargs[f"raw_{i}"] = u_ - 0.000_001  # type: ignore[literal-required]
                            v3_kwargs[f"phys_{i}"] = p_  # type: ignore[literal-required]
                            v3_kwargs[f"phys_{i}"] = p_  # type: ignore[literal-required]

                        conversion = v3b.ChannelConversion(**v3_kwargs)

                    case v4c.CONVERSION_TYPE_TABX:
                        nr = conversion.val_param_nr

                        v3_kwargs = {
                            "ref_param_nr": nr + 1,
                            "unit": unit,
                            "conversion_type": v3c.CONVERSION_TYPE_RTABX,
                        }
                        for i in range(nr):
                            v3_kwargs[f"lower_{i}"] = conversion[f"val_{i}"]  # type: ignore[literal-required]
                            v3_kwargs[f"upper_{i}"] = conversion[f"val_{i}"]  # type: ignore[literal-required]
                            block = conversion.referenced_blocks[f"text_{i}"]
                            if isinstance(block, v4b.ChannelConversion):
                                v3_kwargs[f"text_{i}"] = block.name.encode("latin-1")  # type: ignore[literal-required]
                            else:
                                v3_kwargs[f"text_{i}"] = block  # type: ignore[literal-required]

                        new_conversion = v3b.ChannelConversion(**v3_kwargs)
                        if isinstance(
                            conversion.referenced_blocks["default_addr"],
                            v4b.ChannelConversion,
                        ):
                            default_addr = conversion.referenced_blocks["default_addr"].name.encode("latin-1")
                        else:
                            default_addr = conversion.referenced_blocks["default_addr"]
                        new_conversion.referenced_blocks["default_addr"] = default_addr

                        conversion = new_conversion

                    case v4c.CONVERSION_TYPE_RTABX:
                        nr = conversion.val_param_nr // 2
                        v3_kwargs = {
                            "ref_param_nr": nr + 1,
                            "unit": unit,
                            "conversion_type": v3c.CONVERSION_TYPE_RTABX,
                        }
                        for i in range(nr):
                            v3_kwargs[f"lower_{i}"] = conversion[f"lower_{i}"]  # type: ignore[literal-required]
                            v3_kwargs[f"upper_{i}"] = conversion[f"upper_{i}"]  # type: ignore[literal-required]
                            block = conversion.referenced_blocks[f"text_{i}"]
                            if isinstance(block, v4b.ChannelConversion):
                                v3_kwargs[f"text_{i}"] = block.name.encode("latin-1")  # type: ignore[literal-required]
                            else:
                                v3_kwargs[f"text_{i}"] = block  # type: ignore[literal-required]

                        new_conversion = v3b.ChannelConversion(**v3_kwargs)
                        if isinstance(
                            conversion.referenced_blocks["default_addr"],
                            v4b.ChannelConversion,
                        ):
                            default_addr = conversion.referenced_blocks["default_addr"].name.encode("latin-1")
                        else:
                            default_addr = conversion.referenced_blocks["default_addr"]
                        new_conversion.referenced_blocks["default_addr"] = default_addr

                        conversion = new_conversion

    else:
        if not conversion or isinstance(conversion, v4b.ChannelConversion):
            if copy:
                conversion = deepcopy(conversion)
        else:
            conversion_type = conversion.conversion_type
            unit_str = conversion.unit_field.decode("latin-1").strip(" \r\n\t\0")

            match conversion_type:
                case v3c.CONVERSION_TYPE_NONE:
                    conversion = v4b.ChannelConversion(conversion_type=v4c.CONVERSION_TYPE_NON)

                case v3c.CONVERSION_TYPE_LINEAR:
                    conversion = v4b.ChannelConversion(
                        conversion_type=v4c.CONVERSION_TYPE_LIN,
                        a=conversion.a,
                        b=conversion.b,
                    )

                case v3c.CONVERSION_TYPE_RAT:
                    conversion = v4b.ChannelConversion(
                        conversion_type=v4c.CONVERSION_TYPE_RAT,
                        P1=conversion.P1,
                        P2=conversion.P2,
                        P3=conversion.P3,
                        P4=conversion.P4,
                        P5=conversion.P5,
                        P6=conversion.P6,
                    )

                case v3c.CONVERSION_TYPE_FORMULA:
                    formula = conversion.formula
                    conversion = v4b.ChannelConversion(conversion_type=v4c.CONVERSION_TYPE_ALG, formula=formula)

                case v3c.CONVERSION_TYPE_TAB:
                    v4_kwargs = v4b.ChannelConversionKwargs(conversion_type=v4c.CONVERSION_TYPE_TAB)
                    v4_kwargs["val_param_nr"] = conversion.ref_param_nr * 2
                    for i in range(conversion.ref_param_nr):
                        v4_kwargs[f"raw_{i}"] = conversion[f"raw_{i}"]  # type: ignore[literal-required]
                        v4_kwargs[f"phys_{i}"] = conversion[f"phys_{i}"]  # type: ignore[literal-required]

                    conversion = v4b.ChannelConversion(**v4_kwargs)

                case v3c.CONVERSION_TYPE_TABI:
                    v4_kwargs = v4b.ChannelConversionKwargs(conversion_type=v4c.CONVERSION_TYPE_TABI)
                    v4_kwargs["val_param_nr"] = conversion.ref_param_nr * 2
                    for i in range(conversion.ref_param_nr):
                        v4_kwargs[f"raw_{i}"] = conversion[f"raw_{i}"]  # type: ignore[literal-required]
                        v4_kwargs[f"phys_{i}"] = conversion[f"phys_{i}"]  # type: ignore[literal-required]

                    conversion = v4b.ChannelConversion(**v4_kwargs)

                case v3c.CONVERSION_TYPE_TABX:
                    nr = conversion.ref_param_nr
                    v4_kwargs = {
                        "val_param_nr": nr,
                        "ref_param_nr": nr + 1,
                        "conversion_type": v4c.CONVERSION_TYPE_TABX,
                    }
                    for i in range(nr):
                        v4_kwargs[f"val_{i}"] = conversion[f"param_val_{i}"]  # type: ignore[literal-required]
                        v4_kwargs[f"text_{i}"] = conversion[f"text_{i}"]  # type: ignore[literal-required]

                    conversion = v4b.ChannelConversion(**v4_kwargs)

                case v3c.CONVERSION_TYPE_RTABX:
                    nr = conversion.ref_param_nr - 1
                    v4_kwargs = {
                        "val_param_nr": nr * 2,
                        "ref_param_nr": nr + 1,
                        "conversion_type": v4c.CONVERSION_TYPE_RTABX,
                        "default_addr": typing.cast(bytes, conversion.referenced_blocks["default_addr"]),
                    }
                    for i in range(nr):
                        v4_kwargs[f"lower_{i}"] = conversion[f"lower_{i}"]  # type: ignore[literal-required]
                        v4_kwargs[f"upper_{i}"] = conversion[f"upper_{i}"]  # type: ignore[literal-required]
                        v4_kwargs[f"text_{i}"] = conversion.referenced_blocks[f"text_{i}"]  # type: ignore[literal-required]

                    conversion = v4b.ChannelConversion(**v4_kwargs)

            conversion.unit = unit_str

    return conversion


def inverse_conversion(
    conversion: (
        ChannelConversionType | v3b.ChannelConversionKwargs | v4b.ChannelConversionKwargs | dict[str, object] | None
    ),
) -> v4b.ChannelConversion | None:
    if not conversion:
        return None

    if isinstance(conversion, v3b.ChannelConversion):
        conversion = conversion_transfer(conversion, version=4)

    conversion_dict: v3b.ChannelConversionKwargs | v4b.ChannelConversionKwargs | dict[str, object]
    if not isinstance(conversion, dict):
        conversion_dict = to_dict(conversion)
        if conversion_dict is None:
            return None
    else:
        conversion_dict = conversion

    if "a" in conversion_dict:
        kwargs: v4b.ChannelConversionKwargs = {
            "conversion_type": v4c.CONVERSION_TYPE_LIN,
            "a": 1 / typing.cast(float, conversion_dict["a"]),
            "b": typing.cast(float, conversion_dict["b"]) / typing.cast(float, conversion_dict["a"]),
        }
        conv = v4b.ChannelConversion(**kwargs)

    elif "P1" in conversion_dict:
        a, b, c, d, e, f = (typing.cast(float, conversion_dict[f"P{i}"]) for i in range(1, 7))  # type: ignore[literal-required]

        if e == 0 and f == 0:
            if d == 0 and a == 0:
                conv = None
            else:
                kwargs = {
                    "P1": 0,
                    "P2": e,
                    "P3": -b,
                    "P4": 0,
                    "P5": -d,
                    "P6": a,
                    "conversion_type": v4c.CONVERSION_TYPE_RAT,
                }
                conv = v4b.ChannelConversion(**kwargs)

        elif a == 0 and d == 0:
            if e == 0 and b == 0:
                conv = None
            else:
                kwargs = {
                    "P1": 0,
                    "P2": -f,
                    "P3": c,
                    "P4": 0,
                    "P5": e,
                    "P6": -b,
                    "conversion_type": v4c.CONVERSION_TYPE_RAT,
                }
                conv = v4b.ChannelConversion(**kwargs)

    else:
        conv = None

    return conv


@overload
def from_dict(conversion_dict: v4b.ChannelConversionKwargs | dict[str, object]) -> v4b.ChannelConversion: ...


@overload
def from_dict(conversion_dict: None) -> None: ...


@overload
def from_dict(
    conversion_dict: v4b.ChannelConversionKwargs | dict[str, object] | None,
) -> v4b.ChannelConversion | None: ...


def from_dict(conversion_dict: v4b.ChannelConversionKwargs | dict[str, object] | None) -> v4b.ChannelConversion | None:
    if not conversion_dict:
        conversion = None

    elif "a" in conversion_dict:
        conversion_dict["conversion_type"] = v4c.CONVERSION_TYPE_LIN
        conversion = v4b.ChannelConversion(**conversion_dict)

    elif "formula" in conversion_dict:
        conversion_dict["formula"] = typing.cast(str, conversion_dict["formula"])
        conversion_dict["conversion_type"] = v4c.CONVERSION_TYPE_ALG
        conversion = v4b.ChannelConversion(**conversion_dict)

    elif all(key in conversion_dict for key in [f"P{i}" for i in range(1, 7)]):
        conversion_dict["conversion_type"] = v4c.CONVERSION_TYPE_RAT
        conversion = v4b.ChannelConversion(**conversion_dict)

    elif "raw_0" in conversion_dict and "phys_0" in conversion_dict:
        conversion_dict["conversion_type"] = v4c.CONVERSION_TYPE_TAB
        nr = 0
        while f"phys_{nr}" in conversion_dict:
            nr += 1
        conversion_dict["val_param_nr"] = nr * 2
        if conversion_dict.get("interpolation", False):
            conversion_dict["conversion_type"] = v4c.CONVERSION_TYPE_TABI
        else:
            conversion_dict["conversion_type"] = v4c.CONVERSION_TYPE_TAB
        conversion = v4b.ChannelConversion(**conversion_dict)

    elif "mask_0" in conversion_dict and "text_0" in conversion_dict:
        conversion_dict["conversion_type"] = v4c.CONVERSION_TYPE_BITFIELD
        nr = 0
        while f"text_{nr}" in conversion_dict:
            val = conversion_dict[f"text_{nr}"]  # type: ignore[literal-required]
            if isinstance(val, (bytes, str)):
                partial_conversion: dict[str, object] = {
                    "conversion_type": v4c.CONVERSION_TYPE_RTABX,
                    f"upper_{nr}": conversion_dict[f"upper_{nr}"],  # type: ignore[literal-required]
                    f"lower_{nr}": conversion_dict[f"lower_{nr}"],  # type: ignore[literal-required]
                    f"text_{nr}": (
                        text
                        if isinstance((text := typing.cast(bytes | str, conversion_dict[f"text_{nr}"])), bytes)  # type: ignore[literal-required]
                        else text.encode("utf-8")
                    ),
                    "default": b"",
                }
                conversion_dict[f"text_{nr}"] = from_dict(partial_conversion)  # type: ignore[literal-required]
            elif isinstance(val, dict):
                conversion_dict[f"text_{nr}"] = from_dict(val)  # type: ignore[literal-required]

            nr += 1

        conversion_dict["ref_param_nr"] = nr
        conversion_dict["val_param_nr"] = nr
        conversion = v4b.ChannelConversion(**conversion_dict)

    elif "upper_0" in conversion_dict and "phys_0" in conversion_dict:
        conversion_dict["conversion_type"] = v4c.CONVERSION_TYPE_RTAB
        nr = 0
        while f"phys_{nr}" in conversion_dict:
            nr += 1
        conversion_dict["val_param_nr"] = nr * 3 + 1
        conversion = v4b.ChannelConversion(**conversion_dict)

    elif "val_0" in conversion_dict and "text_0" in conversion_dict:
        conversion_dict["conversion_type"] = v4c.CONVERSION_TYPE_TABX
        nr = 0
        while f"text_{nr}" in conversion_dict:
            val = conversion_dict[f"text_{nr}"]  # type: ignore[literal-required]
            if isinstance(val, str):
                conversion_dict[f"text_{nr}"] = val.encode("utf-8")  # type: ignore[literal-required]
            elif isinstance(val, dict):
                conversion_dict[f"text_{nr}"] = from_dict(val)  # type: ignore[literal-required]
            nr += 1

        val = conversion_dict.get("default_addr", b"")
        if isinstance(val, str):
            conversion_dict["default_addr"] = val.encode("utf-8")
        elif isinstance(val, dict):
            conversion_dict["default_addr"] = from_dict(val)

        conversion_dict["ref_param_nr"] = nr + 1
        conversion = v4b.ChannelConversion(**conversion_dict)

    elif "upper_0" in conversion_dict and "text_0" in conversion_dict:
        conversion_dict["conversion_type"] = v4c.CONVERSION_TYPE_RTABX
        nr = 0
        while f"text_{nr}" in conversion_dict:
            val = conversion_dict[f"text_{nr}"]  # type: ignore[literal-required]
            if isinstance(val, str):
                conversion_dict[f"text_{nr}"] = val.encode("utf-8")  # type: ignore[literal-required]
            elif isinstance(val, dict):
                conversion_dict[f"text_{nr}"] = from_dict(val)  # type: ignore[literal-required]
            nr += 1

        conversion_dict["ref_param_nr"] = nr + 1

        val = conversion_dict.get("default_addr", b"")
        if isinstance(val, str):
            conversion_dict["default_addr"] = val.encode("utf-8")
        elif isinstance(val, dict):
            conversion_dict["default_addr"] = from_dict(val)
        conversion = v4b.ChannelConversion(**conversion_dict)

    elif "default_addr" in conversion_dict:
        conversion_dict["conversion_type"] = v4c.CONVERSION_TYPE_TABX
        val = conversion_dict["default_addr"]
        if isinstance(val, str):
            conversion_dict["default_addr"] = val.encode("utf-8")
        elif isinstance(val, dict):
            conversion_dict["default_addr"] = from_dict(val)
        conversion_dict["ref_param_nr"] = 1
        conversion = v4b.ChannelConversion(**conversion_dict)

    else:
        conversion = v4b.ChannelConversion(conversion_type=v4c.CONVERSION_TYPE_NON)

    return conversion


@overload
def to_dict(conversion: ChannelConversionType) -> dict[str, object]: ...


@overload
def to_dict(conversion: None) -> None: ...


@overload
def to_dict(conversion: ChannelConversionType | None) -> dict[str, object] | None: ...


def to_dict(conversion: ChannelConversionType | None) -> dict[str, object] | None:
    if not conversion:
        return None

    if isinstance(conversion, v3b.ChannelConversion):
        conversion_v4 = conversion_transfer(conversion, version=4)
    else:
        conversion_v4 = conversion

    conversion_type = conversion_v4.conversion_type

    conversion_dict: dict[str, object] = {
        "name": conversion_v4.name,
        "unit": conversion_v4.unit,
        "comment": conversion_v4.comment,
    }

    match conversion_type:
        case v4c.CONVERSION_TYPE_LIN:
            conversion_dict["a"] = conversion_v4.a
            conversion_dict["b"] = conversion_v4.b
            conversion_dict["conversion_type"] = conversion_type

        case v4c.CONVERSION_TYPE_ALG:
            conversion_dict["formula"] = conversion_v4.formula
            conversion_dict["conversion_type"] = conversion_type

        case v4c.CONVERSION_TYPE_RAT:
            conversion_dict.update({key: conversion_v4[key] for key in [f"P{i}" for i in range(1, 7)]})
            conversion_dict["conversion_type"] = conversion_type

        case v4c.CONVERSION_TYPE_TAB | v4c.CONVERSION_TYPE_TABI:
            params = conversion_v4.val_param_nr // 2
            conversion_dict.update({key: conversion_v4[key] for key in [f"phys_{nr}" for nr in range(params)]})
            conversion_dict.update({key: conversion_v4[key] for key in [f"raw_{nr}" for nr in range(params)]})
            conversion_dict["conversion_type"] = conversion_type

        case v4c.CONVERSION_TYPE_RTAB:
            params = (conversion_v4.val_param_nr - 1) // 3
            conversion_dict.update({key: conversion_v4[key] for key in [f"lower_{nr}" for nr in range(params)]})
            conversion_dict.update({key: conversion_v4[key] for key in [f"upper_{nr}" for nr in range(params)]})
            conversion_dict.update({key: conversion_v4[key] for key in [f"phys_{nr}" for nr in range(params)]})
            conversion_dict["conversion_type"] = conversion_type
            conversion_dict["default"] = conversion_v4.default

        case v4c.CONVERSION_TYPE_TABX:
            nr = conversion_v4.ref_param_nr - 1

            conversion_dict["conversion_type"] = conversion_type

            for key, val in conversion_v4.referenced_blocks.items():
                if isinstance(val, str):
                    conversion_dict[key] = val
                elif isinstance(val, v4b.ChannelConversion):
                    conversion_dict[key] = to_dict(val)
                elif val is None:
                    conversion_dict[key] = ""
                else:
                    conversion_dict[key] = val.decode("utf-8", errors="replace")

            for i in range(nr):
                conversion_dict[f"val_{i}"] = conversion_v4[f"val_{i}"]

        case v4c.CONVERSION_TYPE_RTABX:
            nr = conversion_v4.ref_param_nr - 1

            conversion_dict["conversion_type"] = conversion_type

            for key, val in conversion_v4.referenced_blocks.items():
                if isinstance(val, str):
                    conversion_dict[key] = val
                elif isinstance(val, v4b.ChannelConversion):
                    conversion_dict[key] = to_dict(val)
                elif val is None:
                    conversion_dict[key] = ""
                else:
                    conversion_dict[key] = val.decode("utf-8", errors="replace")

            for i in range(nr):
                conversion_dict[f"upper_{i}"] = conversion_v4[f"upper_{i}"]
                conversion_dict[f"lower_{i}"] = conversion_v4[f"lower_{i}"]

        case _:
            return None

    return conversion_dict

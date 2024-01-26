"""
asammdf utility functions for channel conversions
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..types import ChannelConversionType
from . import v2_v3_blocks as v3b
from . import v2_v3_constants as v3c
from . import v4_blocks as v4b
from . import v4_constants as v4c

__all__ = ["conversion_transfer", "from_dict"]


def conversion_transfer(
    conversion: ChannelConversionType, version: int = 3, copy: bool = False
) -> ChannelConversionType:
    """convert between mdf4 and mdf3 channel conversions

    Parameters
    ----------
    conversion : block
        channel conversion
    version : int
        target mdf version
    copy : bool
        return a copy if the input conversion version is the same as the required version

    Returns
    -------
    conversion : block
        channel conversion for specified version

    """

    if version <= 3:
        if conversion is None:
            conversion = v3b.ChannelConversion(conversion_type=v3c.CONVERSION_TYPE_NONE)
        else:
            conversion_type = conversion["conversion_type"]
            if conversion.id == b"CC":
                if copy:
                    conversion = deepcopy(conversion)
            else:
                unit = conversion.unit.strip(" \r\n\t\0").encode("latin-1")

                if conversion_type == v4c.CONVERSION_TYPE_NON:
                    conversion = v3b.ChannelConversion(unit=unit, conversion_type=v3c.CONVERSION_TYPE_NONE)

                elif conversion_type == v4c.CONVERSION_TYPE_LIN:
                    conversion = v3b.ChannelConversion(
                        unit=unit,
                        conversion_type=v3c.CONVERSION_TYPE_LINEAR,
                        a=conversion.a,
                        b=conversion.b,
                    )

                elif conversion_type == v4c.CONVERSION_TYPE_RAT:
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

                elif conversion_type == v4c.CONVERSION_TYPE_TAB:
                    conversion_ = {}
                    conversion_["ref_param_nr"] = conversion.val_param_nr // 2
                    for i in range(conversion.val_param_nr // 2):
                        conversion_[f"raw_{i}"] = conversion[f"raw_{i}"]
                        conversion_[f"phys_{i}"] = conversion[f"phys_{i}"]

                    conversion = v3b.ChannelConversion(
                        unit=unit,
                        conversion_type=v3c.CONVERSION_TYPE_TAB,
                        **conversion_,
                    )

                elif conversion_type == v4c.CONVERSION_TYPE_TABI:
                    conversion_ = {}
                    conversion_["ref_param_nr"] = conversion.val_param_nr // 2
                    for i in range(conversion.val_param_nr // 2):
                        conversion_[f"raw_{i}"] = conversion[f"raw_{i}"]
                        conversion_[f"phys_{i}"] = conversion[f"phys_{i}"]

                    conversion = v3b.ChannelConversion(
                        unit=unit,
                        conversion_type=v3c.CONVERSION_TYPE_TABI,
                        **conversion_,
                    )

                elif conversion_type == v4c.CONVERSION_TYPE_ALG:
                    formula = conversion.formula.replace("X", "X1")
                    conversion = v3b.ChannelConversion(
                        formula=formula,
                        unit=unit,
                        conversion_type=v3c.CONVERSION_TYPE_FORMULA,
                    )

                elif conversion_type == v4c.CONVERSION_TYPE_RTAB:
                    nr = (conversion.val_param_nr - 1) // 3
                    kargs = {
                        "ref_param_nr": nr,
                        "conversion_type": v3c.CONVERSION_TYPE_TABI,
                    }

                    for i in range(nr):
                        l_ = conversion[f"lower_{i}"]
                        u_ = conversion[f"upper_{i}"]
                        p_ = conversion[f"phys_{i}"]
                        kargs[f"raw_{i}"] = l_
                        kargs[f"raw_{i}"] = u_ - 0.000_001
                        kargs[f"phys_{i}"] = p_
                        kargs[f"phys_{i}"] = p_

                    conversion = v3b.ChannelConversion(unit=unit, **kargs)

                elif conversion_type == v4c.CONVERSION_TYPE_TABX:
                    nr = conversion.val_param_nr

                    kargs = {
                        "ref_param_nr": nr + 1,
                        "unit": unit,
                        "conversion_type": v3c.CONVERSION_TYPE_RTABX,
                    }
                    for i in range(nr):
                        kargs[f"lower_{i}"] = conversion[f"val_{i}"]
                        kargs[f"upper_{i}"] = conversion[f"val_{i}"]
                        if isinstance(
                            conversion.referenced_blocks[f"text_{i}"],
                            v4b.ChannelConversion,
                        ):
                            kargs[f"text_{i}"] = conversion.referenced_blocks[f"text_{i}"].name.encode("latin-1")
                        else:
                            kargs[f"text_{i}"] = conversion.referenced_blocks[f"text_{i}"]

                    new_conversion = v3b.ChannelConversion(**kargs)
                    if isinstance(
                        conversion.referenced_blocks["default_addr"],
                        v4b.ChannelConversion,
                    ):
                        default_addr = conversion.referenced_blocks["default_addr"].name.encode("latin-1")
                    else:
                        default_addr = conversion.referenced_blocks["default_addr"]
                    new_conversion.referenced_blocks["default_addr"] = default_addr

                    conversion = new_conversion

                elif conversion_type == v4c.CONVERSION_TYPE_RTABX:
                    nr = conversion.val_param_nr // 2
                    kargs = {
                        "ref_param_nr": nr + 1,
                        "unit": unit,
                        "conversion_type": v3c.CONVERSION_TYPE_RTABX,
                    }
                    for i in range(nr):
                        kargs[f"lower_{i}"] = conversion[f"lower_{i}"]
                        kargs[f"upper_{i}"] = conversion[f"upper_{i}"]
                        if isinstance(
                            conversion.referenced_blocks[f"text_{i}"],
                            v4b.ChannelConversion,
                        ):
                            kargs[f"text_{i}"] = conversion.referenced_blocks[f"text_{i}"].name.encode("latin-1")
                        else:
                            kargs[f"text_{i}"] = conversion.referenced_blocks[f"text_{i}"]

                    new_conversion = v3b.ChannelConversion(**kargs)
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
        if not conversion or conversion.id == b"##CC":
            if copy:
                conversion = deepcopy(conversion)
        else:
            conversion_type = conversion.conversion_type
            unit = conversion.unit_field.decode("latin-1").strip(" \r\n\t\0")

            if conversion_type == v3c.CONVERSION_TYPE_NONE:
                conversion = v4b.ChannelConversion(conversion_type=v4c.CONVERSION_TYPE_NON)

            elif conversion_type == v3c.CONVERSION_TYPE_LINEAR:
                conversion = v4b.ChannelConversion(
                    conversion_type=v4c.CONVERSION_TYPE_LIN,
                    a=conversion.a,
                    b=conversion.b,
                )

            elif conversion_type == v3c.CONVERSION_TYPE_RAT:
                conversion = v4b.ChannelConversion(
                    conversion_type=v4c.CONVERSION_TYPE_RAT,
                    P1=conversion.P1,
                    P2=conversion.P2,
                    P3=conversion.P3,
                    P4=conversion.P4,
                    P5=conversion.P5,
                    P6=conversion.P6,
                )

            elif conversion_type == v3c.CONVERSION_TYPE_FORMULA:
                formula = conversion.formula
                conversion = v4b.ChannelConversion(conversion_type=v4c.CONVERSION_TYPE_ALG, formula=formula)

            elif conversion_type == v3c.CONVERSION_TYPE_TAB:
                conversion_ = {}
                conversion_["val_param_nr"] = conversion.ref_param_nr * 2
                for i in range(conversion.ref_param_nr):
                    conversion_[f"raw_{i}"] = conversion[f"raw_{i}"]
                    conversion_[f"phys_{i}"] = conversion[f"phys_{i}"]

                conversion = v4b.ChannelConversion(conversion_type=v4c.CONVERSION_TYPE_TAB, **conversion_)

            elif conversion_type == v3c.CONVERSION_TYPE_TABI:
                conversion_ = {}
                conversion_["val_param_nr"] = conversion.ref_param_nr * 2
                for i in range(conversion.ref_param_nr):
                    conversion_[f"raw_{i}"] = conversion[f"raw_{i}"]
                    conversion_[f"phys_{i}"] = conversion[f"phys_{i}"]

                conversion = v4b.ChannelConversion(conversion_type=v4c.CONVERSION_TYPE_TABI, **conversion_)

            elif conversion_type == v3c.CONVERSION_TYPE_TABX:
                nr = conversion["ref_param_nr"]
                kargs = {
                    "val_param_nr": nr,
                    "ref_param_nr": nr + 1,
                    "conversion_type": v4c.CONVERSION_TYPE_TABX,
                }
                for i in range(nr):
                    kargs[f"val_{i}"] = conversion[f"param_val_{i}"]
                    kargs[f"text_{i}"] = conversion[f"text_{i}"]

                conversion = v4b.ChannelConversion(**kargs)

            elif conversion_type == v3c.CONVERSION_TYPE_RTABX:
                nr = conversion["ref_param_nr"] - 1
                kargs = {
                    "val_param_nr": nr * 2,
                    "ref_param_nr": nr + 1,
                    "conversion_type": v4c.CONVERSION_TYPE_RTABX,
                    "default_addr": conversion.referenced_blocks["default_addr"],
                }
                for i in range(nr):
                    kargs[f"lower_{i}"] = conversion[f"lower_{i}"]
                    kargs[f"upper_{i}"] = conversion[f"upper_{i}"]
                    kargs[f"text_{i}"] = conversion.referenced_blocks[f"text_{i}"]

                conversion = v4b.ChannelConversion(**kargs)

            conversion.unit = unit

    return conversion


def from_dict(conversion: dict[str, Any]) -> v4b.ChannelConversion:
    conversion = dict(conversion)

    if not conversion:
        conversion = None

    elif "a" in conversion:
        conversion["conversion_type"] = v4c.CONVERSION_TYPE_LIN
        conversion = v4b.ChannelConversion(**conversion)

    elif "formula" in conversion:
        conversion["formula"] = conversion["formula"]
        conversion["conversion_type"] = v4c.CONVERSION_TYPE_ALG
        conversion = v4b.ChannelConversion(**conversion)

    elif all(key in conversion for key in [f"P{i}" for i in range(1, 7)]):
        conversion["conversion_type"] = v4c.CONVERSION_TYPE_RAT
        conversion = v4b.ChannelConversion(**conversion)

    elif "raw_0" in conversion and "phys_0" in conversion:
        conversion["conversion_type"] = v4c.CONVERSION_TYPE_TAB
        nr = 0
        while f"phys_{nr}" in conversion:
            nr += 1
        conversion["val_param_nr"] = nr * 2
        if "interpolation" in conversion:
            conversion["conversion_type"] = v4c.CONVERSION_TYPE_TABI
        else:
            conversion["conversion_type"] = v4c.CONVERSION_TYPE_TAB
        conversion = v4b.ChannelConversion(**conversion)

    elif "mask_0" in conversion and "text_0" in conversion:
        conversion["conversion_type"] = v4c.CONVERSION_TYPE_BITFIELD
        nr = 0
        while f"text_{nr}" in conversion:
            val = conversion[f"text_{nr}"]
            if isinstance(val, (bytes, str)):
                partial_conversion = {
                    "conversion_type": v4c.CONVERSION_TYPE_RTABX,
                    f"upper_{nr}": conversion[f"upper_{nr}"],
                    f"lower_{nr}": conversion[f"lower_{nr}"],
                    f"text_{nr}": (
                        conversion[f"text_{nr}"]
                        if isinstance(conversion[f"text_{nr}"], bytes)
                        else conversion[f"text_{nr}"].encode("utf-8")
                    ),
                    "default": b"",
                }
                conversion[f"text_{nr}"] = from_dict(partial_conversion)
            elif isinstance(val, dict):
                conversion[f"text_{nr}"] = from_dict(val)

            nr += 1

        conversion["ref_param_nr"] = nr
        conversion["val_param_nr"] = nr
        conversion = v4b.ChannelConversion(**conversion)

    elif "upper_0" in conversion and "phys_0" in conversion:
        conversion["conversion_type"] = v4c.CONVERSION_TYPE_RTAB
        nr = 0
        while f"phys_{nr}" in conversion:
            nr += 1
        conversion["val_param_nr"] = nr * 3 + 1
        conversion = v4b.ChannelConversion(**conversion)

    elif "val_0" in conversion and "text_0" in conversion:
        conversion["conversion_type"] = v4c.CONVERSION_TYPE_TABX
        nr = 0
        while f"text_{nr}" in conversion:
            val = conversion[f"text_{nr}"]
            if isinstance(val, str):
                conversion[f"text_{nr}"] = val.encode("utf-8")
            elif isinstance(val, dict):
                conversion[f"text_{nr}"] = from_dict(val)
            nr += 1

        val = conversion.get("default_addr", b"")
        if isinstance(val, str):
            conversion["default_addr"] = val.encode("utf-8")
        elif isinstance(val, dict):
            conversion["default_addr"] = from_dict(val)

        conversion["ref_param_nr"] = nr + 1
        conversion = v4b.ChannelConversion(**conversion)

    elif "upper_0" in conversion and "text_0" in conversion:
        conversion["conversion_type"] = v4c.CONVERSION_TYPE_RTABX
        nr = 0
        while f"text_{nr}" in conversion:
            val = conversion[f"text_{nr}"]
            if isinstance(val, str):
                conversion[f"text_{nr}"] = val.encode("utf-8")
            elif isinstance(val, dict):
                conversion[f"text_{nr}"] = from_dict(val)
            nr += 1

        conversion["ref_param_nr"] = nr + 1

        val = conversion.get("default_addr", b"")
        if isinstance(val, str):
            conversion["default_addr"] = val.encode("utf-8")
        elif isinstance(val, dict):
            conversion["default_addr"] = from_dict(val)
        conversion = v4b.ChannelConversion(**conversion)

    else:
        conversion = v4b.ChannelConversion(conversion_type=v4c.CONVERSION_TYPE_NON)

    return conversion


def to_dict(conversion: ChannelConversionType) -> dict | None:
    if not conversion:
        return None

    if isinstance(conversion, v3b.ChannelConversion):
        conversion = conversion_transfer(conversion, version=4)

    conversion_type = conversion.conversion_type

    conversion_dict = {
        "name": conversion.name,
        "unit": conversion.unit,
        "comment": conversion.comment,
    }

    if conversion_type == v4c.CONVERSION_TYPE_LIN:
        conversion_dict["a"] = conversion.a
        conversion_dict["b"] = conversion.b
        conversion_dict["conversion_type"] = conversion_type

    elif conversion_type == v4c.CONVERSION_TYPE_ALG:
        conversion_dict["formula"] = conversion["formula"]
        conversion_dict["conversion_type"] = conversion_type

    elif conversion_type == v4c.CONVERSION_TYPE_RAT:
        conversion_dict.update({key: conversion[key] for key in [f"P{i}" for i in range(1, 7)]})
        conversion_dict["conversion_type"] = conversion_type

    elif conversion_type in (v4c.CONVERSION_TYPE_TAB, v4c.CONVERSION_TYPE_TABI):
        params = conversion["val_param_nr"] // 2
        conversion_dict.update({key: conversion[key] for key in [f"phys_{nr}" for nr in range(params)]})
        conversion_dict.update({key: conversion[key] for key in [f"raw_{nr}" for nr in range(params)]})
        conversion_dict["conversion_type"] = conversion_type

    elif conversion_type == v4c.CONVERSION_TYPE_RTAB:
        params = (conversion["val_param_nr"] - 1) // 3
        conversion_dict.update({key: conversion[key] for key in [f"lower_{nr}" for nr in range(params)]})
        conversion_dict.update({key: conversion[key] for key in [f"upper_{nr}" for nr in range(params)]})
        conversion_dict.update({key: conversion[key] for key in [f"phys_{nr}" for nr in range(params)]})
        conversion_dict["conversion_type"] = conversion_type
        conversion_dict["default"] = conversion.default

    elif conversion_type == v4c.CONVERSION_TYPE_TABX:
        nr = conversion.ref_param_nr - 1

        conversion_dict["conversion_type"] = conversion_type

        for key, val in conversion.referenced_blocks.items():
            if isinstance(val, str):
                conversion_dict[key] = val
            elif isinstance(val, v4b.ChannelConversion):
                conversion_dict[key] = to_dict(val)
            else:
                conversion_dict[key] = val.decode("utf-8", errors="replace")

        for i in range(nr):
            conversion_dict[f"val_{i}"] = conversion[f"val_{i}"]

    elif conversion_type == v4c.CONVERSION_TYPE_RTABX:
        nr = conversion.ref_param_nr - 1

        conversion_dict["conversion_type"] = conversion_type

        for key, val in conversion.referenced_blocks.items():
            if isinstance(val, str):
                conversion_dict[key] = val
            elif isinstance(val, v4b.ChannelConversion):
                conversion_dict[key] = to_dict(val)
            else:
                conversion_dict[key] = val.decode("utf-8", errors="replace")

        for i in range(nr):
            conversion_dict[f"upper_{i}"] = conversion[f"upper_{i}"]
            conversion_dict[f"lower_{i}"] = conversion[f"lower_{i}"]

    else:
        conversion_dict = None

    return conversion_dict

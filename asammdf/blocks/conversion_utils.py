# -*- coding: utf-8 -*-
"""
asammdf utility functions for channel conversions
"""

from . import v2_v3_constants as v3c
from . import v2_v3_blocks as v3b
from . import v4_constants as v4c
from . import v4_blocks as v4b

__all__ = ["conversion_transfer", "from_dict"]


def conversion_transfer(conversion, version=3):
    """ convert between mdf4 and mdf3 channel conversions

    Parameters
    ----------
    conversion : block
        channel conversion
    version : int
        target mdf version

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
            if conversion["id"] == b"CC":
                pass
            else:
                unit = conversion.unit.strip(" \r\n\t\0").encode("latin-1")

                if conversion_type == v4c.CONVERSION_TYPE_NON:
                    conversion = v3b.ChannelConversion(
                        unit=unit, conversion_type=v3c.CONVERSION_TYPE_NONE
                    )

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
                        kargs[f"text_{i}"] = conversion.referenced_blocks[f"text_{i}"]

                    new_conversion = v3b.ChannelConversion(**kargs)
                    new_conversion.referenced_blocks[
                        "default_addr"
                    ] = conversion.referenced_blocks["default_addr"]

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
                        kargs[f"text_{i}"] = conversion.referenced_blocks[f"text_{i}"]

                    new_conversion = v3b.ChannelConversion(**kargs)
                    new_conversion.referenced_blocks[
                        "default_addr"
                    ] = conversion.referenced_blocks["default_addr"]

                    conversion = new_conversion

    else:
        if conversion is None or conversion["id"] == b"##CC":
            pass
        else:
            conversion_type = conversion["conversion_type"]
            unit = conversion["unit_field"].decode("latin-1").strip(" \r\n\t\0")

            if conversion_type == v3c.CONVERSION_TYPE_NONE:
                conversion = v4b.ChannelConversion(
                    conversion_type=v4c.CONVERSION_TYPE_NON
                )

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
                formula = conversion.formula.replace("X1", "X")
                conversion = v4b.ChannelConversion(
                    conversion_type=v4c.CONVERSION_TYPE_ALG, formula=formula
                )

            elif conversion_type == v3c.CONVERSION_TYPE_TAB:
                conversion_ = {}
                conversion_["val_param_nr"] = conversion.ref_param_nr * 2
                for i in range(conversion.ref_param_nr):
                    conversion_[f"raw_{i}"] = conversion[f"raw_{i}"]
                    conversion_[f"phys_{i}"] = conversion[f"phys_{i}"]

                conversion = v4b.ChannelConversion(
                    conversion_type=v4c.CONVERSION_TYPE_TAB, **conversion_
                )

            elif conversion_type == v3c.CONVERSION_TYPE_TABI:
                conversion_ = {}
                conversion_["val_param_nr"] = conversion.ref_param_nr * 2
                for i in range(conversion.ref_param_nr):
                    conversion_[f"raw_{i}"] = conversion[f"raw_{i}"]
                    conversion_[f"phys_{i}"] = conversion[f"phys_{i}"]

                conversion = v4b.ChannelConversion(
                    conversion_type=v4c.CONVERSION_TYPE_TABI, **conversion_
                )

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


def from_dict(conversion):
    if not conversion:
        conversion = None

    elif "a" in conversion:
        conversion["conversion_type"] = v4c.CONVERSION_TYPE_LIN
        conversion = v4b.ChannelConversion(**conversion)

    elif "formula" in conversion:
        conversion['formula'] = conversion['formula'].replace('X1', 'X')
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
        if conversion['interpolation']:
            conversion['conversion_type'] = v4c.CONVERSION_TYPE_TABI
        else:
            conversion['conversion_type'] = v4c.CONVERSION_TYPE_TAB
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
            nr += 1
        conversion["ref_param_nr"] = nr + 1
        conversion = v4b.ChannelConversion(**conversion)

    elif "upper_0" in conversion and "text_0" in conversion:
        conversion["conversion_type"] = v4c.CONVERSION_TYPE_RTABX
        nr = 0
        while f"text_{nr}" in conversion:
            nr += 1
        conversion["ref_param_nr"] = nr + 1
        conversion = v4b.ChannelConversion(**conversion)

    else:
        conversion = v4b.ChannelConversion(conversion_type=v4c.CONVERSION_TYPE_NON)

    return conversion

from enum import IntEnum
import os
from typing import Final, Literal

from typing_extensions import Any, TypedDict

from .types import StrPath


class IntegerInterpolation(IntEnum):
    REPEAT_PREVIOUS_SAMPLE = 0
    LINEAR_INTERPOLATION = 1
    HYBRID_INTERPOLATION = 2


class FloatInterpolation(IntEnum):
    REPEAT_PREVIOUS_SAMPLE = 0
    LINEAR_INTERPOLATION = 1


class _GlobalOptions(TypedDict):
    read_fragment_size: int
    write_fragment_size: int
    use_display_names: bool
    single_bit_uint_as_bool: bool
    integer_interpolation: IntegerInterpolation
    float_interpolation: FloatInterpolation
    temporary_folder: StrPath | None
    raise_on_multiple_occurrences: bool
    fill_0_for_missing_computation_channels: bool


GLOBAL_OPTIONS: Final[_GlobalOptions] = {
    "read_fragment_size": 256 * 1024 * 1024,
    "write_fragment_size": 4 * 1024 * 1024,
    "use_display_names": True,
    "single_bit_uint_as_bool": False,
    "integer_interpolation": IntegerInterpolation.REPEAT_PREVIOUS_SAMPLE,
    "float_interpolation": FloatInterpolation.LINEAR_INTERPOLATION,
    "temporary_folder": None,
    "raise_on_multiple_occurrences": True,
    "fill_0_for_missing_computation_channels": False,
}

_Opt = Literal[
    "read_fragment_size",
    "write_fragment_size",
    "use_display_names",
    "single_bit_uint_as_bool",
    "integer_interpolation",
    "float_interpolation",
    "temporary_folder",
    "raise_on_multiple_occurrences",
    "fill_0_for_missing_computation_channels",
]


def set_global_option(opt: _Opt, value: Any) -> None:
    if opt not in GLOBAL_OPTIONS:
        raise KeyError(f'Unknown global option "{opt}"')

    if opt == "read_fragment_size":
        GLOBAL_OPTIONS[opt] = int(value)
    elif opt == "write_fragment_size":
        GLOBAL_OPTIONS[opt] = min(int(value), 4 * 1024 * 1024)
    elif opt in (
        "use_display_names",
        "single_bit_uint_as_bool",
        "raise_on_multiple_occurrences",
        "fill_0_for_missing_computation_channels",
    ):
        GLOBAL_OPTIONS[opt] = bool(value)
    elif opt == "integer_interpolation":
        GLOBAL_OPTIONS[opt] = IntegerInterpolation(value)
    elif opt == "float_interpolation":
        GLOBAL_OPTIONS[opt] = FloatInterpolation(value)
    elif opt == "temporary_folder":
        value = value or None
        if value is not None:
            os.makedirs(value, exist_ok=True)
        GLOBAL_OPTIONS[opt] = value


def get_global_option(opt: _Opt) -> object:
    return GLOBAL_OPTIONS[opt]


def get_option(opt: _Opt, instance_options: _GlobalOptions) -> object:
    value: object = instance_options[opt]
    if value is None:
        value = get_global_option(opt)

    return value

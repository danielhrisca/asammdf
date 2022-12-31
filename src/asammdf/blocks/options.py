from enum import IntEnum
import os


class IntegerInterpolation(IntEnum):
    REPEAT_PREVIOUS_SAMPLE = 0
    LINEAR_INTERPOLATION = 1
    HYBRID_INTERPOLATION = 2


class FloatInterpolation(IntEnum):
    REPEAT_PREVIOUS_SAMPLE = 0
    LINEAR_INTERPOLATION = 1


_GLOBAL_OPTIONS = {
    "read_fragment_size": 0,
    "write_fragment_size": 4 * 1024 * 1024,
    "use_display_names": True,
    "single_bit_uint_as_bool": False,
    "integer_interpolation": IntegerInterpolation.REPEAT_PREVIOUS_SAMPLE,
    "float_interpolation": FloatInterpolation.LINEAR_INTERPOLATION,
    "copy_on_get": True,
    "temporary_folder": None,
    "raise_on_multiple_occurrences": True,
    "fill_0_for_missing_computation_channels": False,
}


def set_global_option(opt, value):
    if opt not in _GLOBAL_OPTIONS:
        raise KeyError(f'Unknown global option "{opt}"')

    if opt == "read_fragment_size":
        value = int(value)
    elif opt == "write_fragment_size":
        value = min(int(value), 4 * 1024 * 1024)
    elif opt in (
        "use_display_names",
        "single_bit_uint_as_bool",
        "copy_on_get",
        "raise_on_multiple_occurrences",
        "fill_0_for_missing_computation_channels",
    ):
        value = bool(value)
    elif opt == "integer_interpolation":
        value = IntegerInterpolation(value)
    elif opt == "float_interpolation":
        value = FloatInterpolation(value)
    elif opt == "temporary_folder":
        value = value or None
        if value is not None:
            os.makedirs(value, exist_ok=True)

    _GLOBAL_OPTIONS[opt] = value


def get_global_option(opt):
    return _GLOBAL_OPTIONS[opt]


def get_option(opt, instance_options):
    value = instance_options[opt]
    if value is None:
        value = get_global_option(opt)

    return value

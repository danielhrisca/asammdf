# -*- coding: utf-8 -*-
""" MDF v4 constants """
import re
import struct

MAX_UINT64 = (1 << 64) - 1

DATA_TYPE_UNSIGNED_INTEL = 0
DATA_TYPE_UNSIGNED_MOTOROLA = 1
DATA_TYPE_SIGNED_INTEL = 2
DATA_TYPE_SIGNED_MOTOROLA = 3
DATA_TYPE_REAL_INTEL = 4
DATA_TYPE_REAL_MOTOROLA = 5
DATA_TYPE_STRING_LATIN_1 = 6
DATA_TYPE_STRING_UTF_8 = 7
DATA_TYPE_STRING_UTF_16_LE = 8
DATA_TYPE_STRING_UTF_16_BE = 9
DATA_TYPE_BYTEARRAY = 10
DATA_TYPE_MIME_SAMPLE = 11
DATA_TYPE_MIME_STREAM = 12
DATA_TYPE_CANOPEN_DATE = 13
DATA_TYPE_CANOPEN_TIME = 14

NON_SCALAR_TYPES = {
    DATA_TYPE_BYTEARRAY,
    DATA_TYPE_STRING_UTF_8,
    DATA_TYPE_STRING_LATIN_1,
    DATA_TYPE_STRING_UTF_16_BE,
    DATA_TYPE_STRING_UTF_16_LE,
    DATA_TYPE_CANOPEN_DATE,
    DATA_TYPE_CANOPEN_TIME,
}

STRING_TYPES = {
    DATA_TYPE_STRING_UTF_8,
    DATA_TYPE_STRING_LATIN_1,
    DATA_TYPE_STRING_UTF_16_BE,
    DATA_TYPE_STRING_UTF_16_LE,
}

SIGNAL_TYPE_SCALAR = 0
SIGNAL_TYPE_STRING = 1
SIGNAL_TYPE_CANOPEN = 2
SIGNAL_TYPE_STRUCTURE_COMPOSITION = 3
SIGNAL_TYPE_ARRAY = 4
SIGNAL_TYPE_BYTEARRAY = 5

SIGNED_INT = {DATA_TYPE_SIGNED_INTEL, DATA_TYPE_SIGNED_MOTOROLA}
STANDARD_INT_SIZES = {8, 16, 32, 64}

CHANNEL_TYPE_VALUE = 0
CHANNEL_TYPE_VLSD = 1
CHANNEL_TYPE_MASTER = 2
CHANNEL_TYPE_VIRTUAL_MASTER = 3
CHANNEL_TYPE_SYNC = 4
CHANNEL_TYPE_MLSD = 5
CHANNEL_TYPE_VIRTUAL = 6

SYNC_TYPE_NONE = 0
SYNC_TYPE_TIME = 1
SYNC_TYPE_ANGLE = 2
SYNC_TYPE_DISTANCE = 3
SYNC_TYPE_INDEX = 4

CHANNEL_TYPE_TO_DESCRIPTION = {
    CHANNEL_TYPE_VALUE: "value",
    CHANNEL_TYPE_VLSD: "VLSD",
    CHANNEL_TYPE_MASTER: "master",
    CHANNEL_TYPE_VIRTUAL_MASTER: "virtual master",
    CHANNEL_TYPE_SYNC: "sync",
    CHANNEL_TYPE_MLSD: "MLSD",
    CHANNEL_TYPE_VIRTUAL: "virtual",
}

MASTER_TYPES = {CHANNEL_TYPE_MASTER, CHANNEL_TYPE_VIRTUAL_MASTER}
VIRTUAL_TYPES = {CHANNEL_TYPE_VIRTUAL, CHANNEL_TYPE_VIRTUAL_MASTER}

CONVERSION_TYPE_NON = 0
CONVERSION_TYPE_LIN = 1
CONVERSION_TYPE_RAT = 2
CONVERSION_TYPE_ALG = 3
CONVERSION_TYPE_TABI = 4
CONVERSION_TYPE_TAB = 5
CONVERSION_TYPE_RTAB = 6
CONVERSION_TYPE_TABX = 7
CONVERSION_TYPE_RTABX = 8
CONVERSION_TYPE_TTAB = 9
CONVERSION_TYPE_TRANS = 10

CONV_RAT_TEXT = "(P1 * X**2 + P2 * X + P3) / (P4 * X**2 + P5 * X + P6)"

TABULAR_CONVERSIONS = {
    CONVERSION_TYPE_TABX,
    CONVERSION_TYPE_RTABX,
    CONVERSION_TYPE_TTAB,
}

CONVERSIONS_WITH_TEXTS = {
    CONVERSION_TYPE_ALG,
    CONVERSION_TYPE_RTABX,
    CONVERSION_TYPE_TABX,
    CONVERSION_TYPE_TRANS,
    CONVERSION_TYPE_TTAB,
}

CONVERSION_GROUP_1 = {CONVERSION_TYPE_NON, CONVERSION_TYPE_TRANS, CONVERSION_TYPE_TTAB}

CONVERSION_GROUP_2 = {
    CONVERSION_TYPE_LIN,
    CONVERSION_TYPE_RAT,
    CONVERSION_TYPE_ALG,
    CONVERSION_TYPE_TABI,
    CONVERSION_TYPE_TAB,
    CONVERSION_TYPE_RTAB,
}

CA_TYPE_ARRAY = 0
CA_TYPE_SCALE_AXIS = 1
CA_TYPE_LOOKUP = 2
CA_STORAGE_TYPE_CN_TEMPLATE = 0

SOURCE_OTHER = 0
SOURCE_ECU = 1
SOURCE_BUS = 2
SOURCE_IO = 3
SOURCE_TOOL = 4
SOURCE_USER = 5

BUS_TYPE_NONE = 0
BUS_TYPE_OTHER = 1
BUS_TYPE_CAN = 2
BUS_TYPE_LIN = 3
BUS_TYPE_MOST = 4
BUS_TYPE_FLEXRAY = 5
BUS_TYPE_K_LINE = 6
BUS_TYPE_ETHERNET = 7
BUS_TYPE_USB = 8

EVENT_TYPE_RECORDING = 0
EVENT_TYPE_RECORDING_INTERRUPT = 1
EVENT_TYPE_ACQUISITION_INTERRUPT = 2
EVENT_TYPE_START_RECORDING_TRIGGER = 3
EVENT_TYPE_STOP_RECORDING_TRIGGER = 4
EVENT_TYPE_TRIGGER = 5
EVENT_TYPE_MARKER = 6

EVENT_SYNC_TYPE_S = 1
EVENT_SYNC_TYPE_RAD = 2
EVENT_SYNC_TYPE_M = 3
EVENT_SYNC_TYPE_INDEX = 4

EVENT_RANGE_TYPE_POINT = 0
EVENT_RANGE_TYPE_BEGINNING = 1
EVENT_RANGE_TYPE_END = 2

EVENT_CAUSE_OTHER = 0
EVENT_CAUSE_ERROR = 1
EVENT_CAUSE_TOOL = 2
EVENT_CAUSE_SCRIPT = 3
EVENT_CAUSE_USER = 4

SEEK_START = 0
SEEK_REL = 1
SEEK_END = 2

TIME_CH_SIZE = 8
SI_BLOCK_SIZE = 56
FH_BLOCK_SIZE = 56
DG_BLOCK_SIZE = 64
HD_BLOCK_SIZE = 104
CN_BLOCK_SIZE = 160
CG_BLOCK_SIZE = 104
COMMON_SIZE = 24
CC_NONE_BLOCK_SIZE = 80
CC_ALG_BLOCK_SIZE = 88
CC_LIN_BLOCK_SIZE = 96
AT_COMMON_SIZE = 96
DZ_COMMON_SIZE = 48
CC_COMMON_BLOCK_SIZE = 80
HL_BLOCK_SIZE = 40
IDENTIFICATION_BLOCK_SIZE = 64
HEADER_BLOCK_SIZE = 104
SR_BLOCK_SIZE = 64

FLAG_ALL_SAMPLES_VALID = 1
FLAG_INVALIDATION_BIT_VALID = 2

FLAG_PRECISION = 1
FLAG_PHY_RANGE_OK = 1 << 4
FLAG_VAL_RANGE_OK = 1 << 3
FLAG_AT_EMBEDDED = 1
FLAG_AT_COMPRESSED_EMBEDDED = 2
FLAG_AT_MD5_VALID = 4
FLAG_DZ_DEFLATE = 0
FLAG_DZ_TRANPOSED_DEFLATE = 1
FLAG_CA_FIXED_AXIS = 1 << 5
FLAG_CA_AXIS = 1 << 4
FLAG_CA_INVERSE_LAYOUT = 1 << 6
FLAG_DL_EQUAL_LENGHT = 1
FLAG_EV_POST_PROCESSING = 1

FLAG_CG_VLSD = 1
FLAG_CG_BUS_EVENT = 1 << 1
FLAG_CG_PLAIN_BUS_EVENT = 1 << 2

FLAG_CN_ALL_INVALID = 1
FLAG_CN_INVALIDATION_PRESENT = 1 << 1
FLAG_CN_PRECISION = 1 << 2
FLAG_CN_VALUE_RANGE = 1 << 3
FLAG_CN_LIMIT_RANGE = 1 << 4
FLAG_CN_EXTENDED_LIMIT_RANGE = 1 << 5
FLAG_CN_DISCRETE = 1 << 6
FLAG_CN_CALIBRATION = 1 << 7
FLAG_CN_CALCULATED = 1 << 8
FLAG_CN_VIRTUAL = 1 << 9
FLAG_CN_BUS_EVENT = 1 << 10
FLAG_CN_MONOTONOUS = 1 << 11
FLAG_CN_DEFAULT_X = 1 << 12

FLAG_CC_PRECISION = 1
FLAG_CC_RANGE = 1 << 1

# data location
LOCATION_ORIGINAL_FILE = 0
LOCATION_TEMPORARY_FILE = 1
LOCATION_MEMORY = 2

# data block type
DT_BLOCK = 0
DZ_BLOCK_DEFLATE = 1
DZ_BLOCK_TRANSPOSED = 2
BLOSC_BLOCK = 3
FMT_CHANNEL = "<4sI2Q{}Q4B4I2BH6d"
FMT_CHANNEL_PARAMS = "<4B4I2BH6d"

FMT_SIMPLE_CHANNEL = "<4sI10Q4B4I2BH6d"
KEYS_SIMPLE_CHANNEL = (
    "id",
    "reserved0",
    "block_len",
    "links_nr",
    "next_ch_addr",
    "component_addr",
    "name_addr",
    "source_addr",
    "conversion_addr",
    "data_block_addr",
    "unit_addr",
    "comment_addr",
    "channel_type",
    "sync_type",
    "data_type",
    "bit_offset",
    "byte_offset",
    "bit_count",
    "flags",
    "pos_invalidation_bit",
    "precision",
    "reserved1",
    "attachment_nr",
    "min_raw_value",
    "max_raw_value",
    "lower_limit",
    "upper_limit",
    "lower_ext_limit",
    "upper_ext_limit",
)
FMT_SIMPLE_CHANNEL_PARAMS = "<8Q4B4I2BH6d"
SIMPLE_CHANNEL_PARAMS_u = struct.Struct(FMT_SIMPLE_CHANNEL_PARAMS).unpack
SIMPLE_CHANNEL_PARAMS_uf = struct.Struct(FMT_SIMPLE_CHANNEL_PARAMS).unpack_from
SIMPLE_CHANNEL_PACK = struct.Struct(FMT_SIMPLE_CHANNEL).pack

FMT_TEXT_BLOCK = "<4sIQQ{}s"
KEYS_TEXT_BLOCK = ("id", "reserved0", "block_len", "links_nr", "text")

FMT_SOURCE_INFORMATION = "<4sI5Q3B5s"
KEYS_SOURCE_INFORMATION = (
    "id",
    "reserved0",
    "block_len",
    "links_nr",
    "name_addr",
    "path_addr",
    "comment_addr",
    "source_type",
    "bus_type",
    "flags",
    "reserved1",
)
SOURCE_INFORMATION_PACK = struct.Struct(FMT_SOURCE_INFORMATION).pack

FMT_CHANNEL_GROUP = "<4sI10Q2H3I"
KEYS_CHANNEL_GROUP = (
    "id",
    "reserved0",
    "block_len",
    "links_nr",
    "next_cg_addr",
    "first_ch_addr",
    "acq_name_addr",
    "acq_source_addr",
    "first_sample_reduction_addr",
    "comment_addr",
    "record_id",
    "cycles_nr",
    "flags",
    "path_separator",
    "reserved1",
    "samples_byte_nr",
    "invalidation_bytes_nr",
)
CHANNEL_GROUP_u = struct.Struct(FMT_CHANNEL_GROUP).unpack
CHANNEL_GROUP_uf = struct.Struct(FMT_CHANNEL_GROUP).unpack_from
CHANNEL_GROUP_p = struct.Struct(FMT_CHANNEL_GROUP).pack

FMT_DATA_BLOCK = "<4sI2Q{}s"
KEYS_DATA_BLOCK = ("id", "reserved0", "block_len", "links_nr", "data")

FMT_COMMON = "<4sI2Q"
COMMON_u = struct.Struct(FMT_COMMON).unpack
COMMON_uf = struct.Struct(FMT_COMMON).unpack_from
COMMON_p = struct.Struct(FMT_COMMON).pack

FMT_FILE_HISTORY = "<4sI5Q2HB3s"
KEYS_FILE_HISTORY = (
    "id",
    "reserved0",
    "block_len",
    "links_nr",
    "next_fh_addr",
    "comment_addr",
    "abs_time",
    "tz_offset",
    "daylight_save_time",
    "time_flags",
    "reserved1",
)

FMT_DATA_GROUP = "<4sI6QB7s"
KEYS_DATA_GROUP = (
    "id",
    "reserved0",
    "block_len",
    "links_nr",
    "next_dg_addr",
    "first_cg_addr",
    "data_block_addr",
    "comment_addr",
    "record_id_len",
    "reserved1",
)
DATA_GROUP_u = struct.Struct(FMT_DATA_GROUP).unpack
DATA_GROUP_uf = struct.Struct(FMT_DATA_GROUP).unpack_from
DATA_GROUP_p = struct.Struct(FMT_DATA_GROUP).pack

FMT_DATA_LIST = "<4sI2Q{}QB3sIQ"

FMT_CONVERSION_NONE = "<4sI6Q2B3H2d"
KEYS_CONVERSION_NONE = (
    "id",
    "reserved0",
    "block_len",
    "links_nr",
    "name_addr",
    "unit_addr",
    "comment_addr",
    "inv_conv_addr",
    "conversion_type",
    "precision",
    "flags",
    "ref_param_nr",
    "val_param_nr",
    "min_phy_value",
    "max_phy_value",
)
CONVERSION_NONE_PACK = struct.Struct(FMT_CONVERSION_NONE).pack
FMT_CONVERSION_NONE_INIT = "<4Q2B3H2d"
CONVERSION_NONE_INIT_u = struct.Struct(FMT_CONVERSION_NONE_INIT).unpack
CONVERSION_NONE_INIT_uf = struct.Struct(FMT_CONVERSION_NONE_INIT).unpack_from

FMT_CONVERSION_LINEAR = FMT_CONVERSION_NONE + "2d"
CONVERSION_LINEAR_PACK = struct.Struct(FMT_CONVERSION_LINEAR).pack
KEYS_CONVERSION_LINEAR = KEYS_CONVERSION_NONE + ("b", "a")
FMT_CONVERSION_LINEAR_INIT = "<4Q2B3H4d"
CONVERSION_LINEAR_INIT_u = struct.Struct(FMT_CONVERSION_LINEAR_INIT).unpack
CONVERSION_LINEAR_INIT_uf = struct.Struct(FMT_CONVERSION_LINEAR_INIT).unpack_from

FMT_CONVERSION_ALGEBRAIC = "<4sI7Q2B3H2d"
CONVERSION_ALGEBRAIC_PACK = struct.Struct(FMT_CONVERSION_ALGEBRAIC).pack
KEYS_CONVERSION_ALGEBRAIC = (
    "id",
    "reserved0",
    "block_len",
    "links_nr",
    "name_addr",
    "unit_addr",
    "comment_addr",
    "inv_conv_addr",
    "formula_addr",
    "conversion_type",
    "precision",
    "flags",
    "ref_param_nr",
    "val_param_nr",
    "min_phy_value",
    "max_phy_value",
)
FMT_CONVERSION_ALGEBRAIC_INIT = "<5Q2B3H2d"

FMT_CONVERSION_RAT = FMT_CONVERSION_NONE + "6d"
CONVERSION_RAT_PACK = struct.Struct(FMT_CONVERSION_RAT).pack
KEYS_CONVERSION_RAT = KEYS_CONVERSION_NONE + ("P1", "P2", "P3", "P4", "P5", "P6")

FMT_CONVERSION_RAT_INIT = "<4Q2B3H8d"

FMT_CONVERSION_RAT_INIT = "<4Q2B3H8d"

FMT_HEADER_BLOCK = "<4sI9Q2H4B2Q"
FMT_IDENTIFICATION_BLOCK = "<8s8s8s4sH30s2H"

KEYS_HEADER_BLOCK = (
    "id",
    "reserved0",
    "block_len",
    "links_nr",
    "first_dg_addr",
    "file_history_addr",
    "channel_tree_addr",
    "first_attachment_addr",
    "first_event_addr",
    "comment_addr",
    "abs_time",
    "tz_offset",
    "daylight_save_time",
    "time_flags",
    "time_quality",
    "flags",
    "reserved1",
    "start_angle",
    "start_distance",
)

KEYS_IDENTIFICATION_BLOCK = (
    "file_identification",
    "version_str",
    "program_identification",
    "reserved0",
    "mdf_version",
    "reserved1",
    "unfinalized_standard_flags",
    "unfinalized_custom_flags",
)

FMT_AT_COMMON = "<4sI6Q2HI16s2Q"
KEYS_AT_BLOCK = (
    "id",
    "reserved0",
    "block_len",
    "links_nr",
    "next_at_addr",
    "file_name_addr",
    "mime_addr",
    "comment_addr",
    "flags",
    "creator_index",
    "reserved1",
    "md5_sum",
    "original_size",
    "embedded_size",
    "embedded_data",
)
AT_COMMON_u = struct.Struct(FMT_AT_COMMON).unpack
AT_COMMON_uf = struct.Struct(FMT_AT_COMMON).unpack_from

FMT_DZ_COMMON = "<4sI2Q2s2BI2Q"
KEYS_DZ_BLOCK = (
    "id",
    "reserved0",
    "block_len",
    "links_nr",
    "original_type",
    "zip_type",
    "reserved1",
    "param",
    "original_size",
    "zip_size",
    "data",
)
DZ_COMMON_u = struct.Struct(FMT_DZ_COMMON).unpack
DZ_COMMON_uf = struct.Struct(FMT_DZ_COMMON).unpack_from
DZ_COMMON_p = struct.Struct(FMT_DZ_COMMON).pack

FMT_HL_BLOCK = "<4sI3QHB5s"
KEYS_HL_BLOCK = (
    "id",
    "reserved0",
    "block_len",
    "links_nr",
    "first_dl_addr",
    "flags",
    "zip_type",
    "reserved1",
)

FMT_EVENT_PARAMS = "<5B3sI2HQd"
FMT_EVENT = "<4sI2Q{}Q5B3sI2HQd"

FMT_SR_BLOCK = "<4sI5Qd2B6s"
KEYS_SR_BLOCK = (
    "id",
    "reserved0",
    "block_len",
    "links_nr",
    "next_sr_addr",
    "data_block_addr",
    "cycles_nr",
    "interval",
    "sync_type",
    "flags",
    "reserved1",
)

ASAM_XML_NAMESPACE = "{http://www.asam.net/mdf/v4}"

CAN_ID_PATTERN = re.compile(r"((?i)can)(?P<id>\d+)")
CAN_DATA_FRAME_PATTERN = re.compile(r"((?i)CAN_DataFrame)_(?P<id>\d+)")

CN_COMMENT_TEMPLATE = """<CNcomment>
<TX>{}</TX>
<names>
    <display>{}</display>
</names>
</CNcomment>"""

HD_COMMENT_TEMPLATE = """<HDcomment>
<TX>{}</TX>
<common_properties>
    <e name="author">{}</e>
    <e name="department">{}</e>
    <e name="project">{}</e>
    <e name="subject">{}</e>
</common_properties>
</HDcomment>"""

CANOPEN_TIME_FIELDS = ("ms", "days")
CANOPEN_DATE_FIELDS = (
    "ms",
    "min",
    "hour",
    "day",
    "month",
    "year",
    "summer_time",
    "day_of_week",
)

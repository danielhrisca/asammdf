# -*- coding: utf-8 -*-
"""
MDF v4 constants
"""

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

CHANNEL_TYPE_VALUE = 0
CHANNEL_TYPE_VLSD = 1
CHANNEL_TYPE_MASTER = 2
CHANNEL_TYPE_VIRTUAL_MASTER = 3
CHANNEL_TYPE_SYNC = 4
CHANNEL_TYPE_MLSD = 5
CHANNEL_TYPE_VIRTUAL = 6

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

SOURCE_ECU = 1
SOURCE_BUS = 2
SOURCE_IO = 3
SOURCE_TOOL = 4

BUS_TYPE_NONE = 0
BUS_TYPE_CAN = 2
BUS_TYPE_FLEXRAY = 5

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

FLAG_PRECISION = 1
FLAG_PHY_RANGE_OK = 2
FLAG_VAL_RANGE_OK = 8
FLAG_AT_EMBEDDED = 1
FLAG_AT_COMPRESSED_EMBEDDED = 2
FLAG_AT_MD5_VALID = 4
FLAG_DZ_DEFLATE = 0
FLAG_DZ_TRANPOSED_DEFLATE = 1

FMT_CHANNEL = '<4sI10Q4B4I2BH6d'
KEYS_CHANNEL = ('id',
                'reserved0',
                'block_len',
                'links_nr',
                'next_ch_addr',
                'component_addr',
                'name_addr',
                'source_addr',
                'conversion_addr',
                'data_block_addr',
                'unit_addr',
                'comment_addr',
                'channel_type',
                'sync_type',
                'data_type',
                'bit_offset',
                'byte_offset',
                'bit_count',
                'flags',
                'pos_invalidation_bit',
                'precision',
                'reserved1',
                'attachment_nr',
                'min_raw_value',
                'max_raw_value',
                'lower_limit',
                'upper_limit',
                'lower_ext_limit',
                'upper_ext_limit')

FMT_TEXT_BLOCK = '<4sIQQ{}s'
KEYS_TEXT_BLOCK = ('id',
                   'reserved0',
                   'block_len',
                   'links_nr',
                   'text')

FMT_SOURCE_INFORMATION = '<4sI5Q3B5s'
KEYS_SOURCE_INFORMATION = ('id',
                           'reserved0',
                           'block_len',
                           'links_nr',
                           'name_addr',
                           'path_addr',
                           'comment_addr',
                           'source_type',
                           'bus_type',
                           'flags',
                           'reserved1')

FMT_CHANNEL_GROUP = '<4sI10Q2H3I'
KEYS_CHANNEL_GROUP = ('id',
                      'reserved0',
                      'block_len',
                      'links_nr',
                      'next_cg_addr',
                      'first_ch_addr',
                      'acq_name_addr',
                      'acq_source_addr',
                      'first_sample_reduction_addr',
                      'comment_addr',
                      'record_id',
                      'cycles_nr',
                      'flags',
                      'path_separator',
                      'reserved1',
                      'samples_byte_nr',
                      'invalidation_bytes_nr')

FMT_DATA_BLOCK = '<4sI2Q{}s'
KEYS_DATA_BLOCK = ('id',
                   'reserved0',
                   'block_len',
                   'links_nr',
                   'data')

FMT_COMMON = '<4sI2Q'

FMT_FILE_HISTORY = '<4sI5Q2HB3s'
KEYS_FILE_HISTORY = ('id',
                     'reserved0',
                     'block_len',
                     'links_nr',
                     'next_fh_addr',
                     'comment_addr',
                     'abs_time',
                     'tz_offset',
                     'daylight_save_time',
                     'time_flags',
                     'reserved1')

FMT_DATA_GROUP = '<4sI6QB7s'
KEYS_DATA_GROUP = ('id',
                   'reserved0',
                   'block_len',
                   'links_nr',
                   'next_dg_addr',
                   'first_cg_addr',
                   'data_block_addr',
                   'comment_addr',
                   'record_id_len',
                   'reserved1')

FMT_DATA_LIST = '<4sIQQ{}QB3sIQ'

FMT_CONVERSION_NONE = '<4sI6Q2B3H2d'
KEYS_CONVERSION_NONE = ('id',
                        'reserved0',
                        'block_len',
                        'links_nr',
                        'name_addr',
                        'unit_addr',
                        'comment_addr',
                        'inv_conv_addr',
                        'conversion_type',
                        'precision',
                        'flags',
                        'ref_param_nr',
                        'val_param_nr',
                        'min_phy_value',
                        'max_phy_value')

FMT_CONVERSION_LINEAR = FMT_CONVERSION_NONE + '2d'
KEYS_CONVERSION_LINEAR = KEYS_CONVERSION_NONE + ('b', 'a')

FMT_CONVERSION_ALGEBRAIC = FMT_CONVERSION_NONE + 'Q'
KEYS_CONVERSION_ALGEBRAIC = KEYS_CONVERSION_NONE + ('formula_addr',)

FMT_CONVERSION_RAT = FMT_CONVERSION_NONE + '6d'
KEYS_CONVERSION_RAT = KEYS_CONVERSION_NONE + ('P1', 'P2', 'P3', 'P4', 'P5', 'P6')

FMT_HEADER_BLOCK = '<4sI9Q2H4B2Q'
FMT_IDENTIFICATION_BLOCK = '<8s8s8s5H26s2H'
IDENTIFICATION_BLOCK_SIZE = 64
HEADER_BLOCK_SIZE = 104
KEYS_HEADER_BLOCK = ('id',
                     'reserved3',
                     'block_len',
                     'links_nr',
                     'first_dg_addr',
                     'file_history_addr',
                     'channel_tree_addr',
                     'first_attachment_addr',
                     'first_event_addr',
                     'comment_addr',
                     'abs_time',
                     'tz_offset',
                     'daylight_save_time',
                     'time_flags',
                     'time_quality',
                     'flags',
                     'reserved4',
                     'start_angle',
                     'start_distance')
KEYS_IDENTIFICATION_BLOCK = ('file_identification',
                             'version_str',
                             'program_identification',
                             'reserved0',
                             'reserved1',
                             'mdf_version',
                             'reserved2',
                             'check_block',
                             'fill',
                             'unfinalized_standard_flags',
                             'unfinalized_custom_flags')

FMT_AT_COMMON = '<4sI6Q2HI16s2Q'
KEYS_AT_BLOCK = ('id',
                 'reserved0',
                 'block_len',
                 'links_nr',
                 'next_at_addr',
                 'file_name_addr',
                 'mime_addr',
                 'comment_addr',
                 'flags',
                 'creator_index',
                 'reserved1',
                 'md5_sum',
                 'original_size',
                 'embedded_size',
                 'embedded_data')

FMT_DZ_COMMON = '<4sI2Q2s2BI2Q'
KEYS_DZ_BLOCK = ('id',
                 'reserved0',
                 'block_len',
                 'links_nr',
                 'original_type',
                 'zip_type',
                 'reserved1',
                 'param',
                 'original_size',
                 'zip_size',
                 'data')

FMT_HL_BLOCK = '<4sI3QHB5s'
KEYS_HL_BLOCK = ('id',
                 'reserved0',
                 'block_len',
                 'links_nr',
                 'first_dl_addr',
                 'flags',
                 'zip_type',
                 'reserved1')


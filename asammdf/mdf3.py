"""
ASAM MDF version 3 file format module

"""

import time
import os
import itertools

from struct import unpack, pack, iter_unpack
from collections import defaultdict


from numpy import (interp, linspace, dtype, amin, amax,
                   array, searchsorted, log, exp, clip)
from numpy.core.records import fromstring, fromarrays
from numexpr import evaluate

try:
    from blosc import compress, decompress
except ImportError:
    from zlib import compress, decompress


conversions = {'CONVERSION_TYPE_NONE' : 65535,
               'CONVERSION_TYPE_LINEAR' : 0,
               'CONVERSION_TYPE_TABI' : 1,
               'CONVERSION_TYPE_TABX' : 2,
               'CONVERSION_TYPE_POLY' : 6,
               'CONVERSION_TYPE_EXPO' : 7,
               'CONVERSION_TYPE_LOGH' : 8,
               'CONVERSION_TYPE_RAT' : 9,
               'CONVERSION_TYPE_FORMULA' : 10,
               'CONVERSION_TYPE_VTAB' : 11,
               'CONVERSION_TYPE_VTABR' : 12}

BYTE_ORDER_INTEL = 0
BYTE_ORDER_MOTOROLA = 1

DATA_TYPE_UNSIGNED = 0
DATA_TYPE_SIGNED = 1
DATA_TYPE_FLOAT = 2
DATA_TYPE_DOUBLE = 3
DATA_TYPE_STRING = 7
DATA_TYPE_BYTEARRAY = 8
DATA_TYPE_UNSIGNED_INTEL = 13
DATA_TYPE_UNSIGNED_MOTOROLA = 9
DATA_TYPE_SIGNED_INTEL = 14
DATA_TYPE_SIGNED_MOTOROLA = 10
DATA_TYPE_FLOAT_INTEL = 15
DATA_TYPE_FLOAT_MOTOROLA = 11
DATA_TYPE_DOUBLE_INTEL = 16
DATA_TYPE_DOUBLE_MOTOROLA = 12

CHANNEL_TYPE_VALUE = 0
CHANNEL_TYPE_MASTER = 1

CONVERSION_TYPE_NONE = 65535
CONVERSION_TYPE_LINEAR = 0
CONVERSION_TYPE_TABI = 1
CONVERSION_TYPE_TABX = 2
CONVERSION_TYPE_POLY = 6
CONVERSION_TYPE_EXPO = 7
CONVERSION_TYPE_LOGH = 8
CONVERSION_TYPE_RAT = 9
CONVERSION_TYPE_FORMULA = 10
CONVERSION_TYPE_VTAB = 11
CONVERSION_TYPE_VTABR = 12

SOURCE_ECU = 2
SOURCE_VECTOR = 19

BUS_TYPE_NONE = 0
BUS_TYPE_CAN = 2
BUS_TYPE_FLEXRAY = 5

SEEK_START = 0
SEEK_REL = 1
SEEK_END = 2

HEADER_COMMON_FMT = '<2sH3IH10s8s32s32s32s32s'
HEADER_COMMON_SIZE = 164
HEADER_POST_320_EXTRA_FMT = 'Q2H32s'
HEADER_POST_320_EXTRA_SIZE = 44

HEADER_COMMON_KEYS = ('id',
                      'block_len',
                      'first_dg_addr',
                      'comment_addr',
                      'program_addr',
                      'dg_nr',
                      'date',
                      'time',
                      'author',
                      'organization',
                      'project',
                      'subject')

HEADER_POST_320_EXTRA_KEYS = ('abs_time',
                              'tz_offset',
                              'time_quality',
                              'timer_identification')

ID_FMT = '<8s8s8s4H2s26s2H'
ID_KEYS = ('file_identification',
           'version_str',
           'program_identification',
           'byte_order',
           'float_format',
           'mdf_version',
           'code_page',
           'reserved0',
           'reserved1',
           'unfinalized_standard_flags',
           'unfinalized_custom_flags')
ID_BLOCK_SIZE = 64

TIME_FAC = 10 ** -9
TIME_CH_SIZE = 8
CE_BLOCK_SIZE = 128
FH_BLOCK_SIZE = 56
DG31_BLOCK_SIZE = 24
DG32_BLOCK_SIZE = 28
HD_BLOCK_SIZE = 104
CN_BLOCK_SIZE = 228
CG_BLOCK_SIZE = 26
CG33_BLOCK_SIZE = 30
DT_BLOCK_SIZE = 24
CC_COMMON_BLOCK_SIZE = 46
CC_ALG_BLOCK_SIZE = 88
CC_LIN_BLOCK_SIZE = 62
CC_POLY_BLOCK_SIZE = 94
CC_EXPO_BLOCK_SIZE = 102
CC_FORMULA_BLOCK_SIZE = 304
SR_BLOCK_SIZE = 156

FLAG_PRECISION = 1
FLAG_PHY_RANGE_OK = 2
FLAG_VAL_RANGE_OK = 8


FMT_CHANNEL = '<2sH5IH32s128s4H3d2IH'
KEYS_CHANNEL = ('id',
                'block_len',
                'next_ch_addr',
                'conversion_addr',
                'source_depend_addr',
                'ch_depend_addr',
                'comment_addr',
                'channel_type',
                'short_name',
                'description',
                'start_offset',
                'bit_count',
                'data_type',
                'range_flag',
                'min_raw_value',
                'max_raw_value',
                'sampling_rate',
                'long_name_addr',
                'display_name_addr',
                'aditional_byte_offset')

FMT_CHANNEL_GROUP = '<2sH3I3HI'
KEYS_CHANNEL_GROUP = ('id',
                      'block_len',
                      'next_cg_addr',
                      'first_ch_addr',
                      'comment_addr',
                      'record_id',
                      'ch_nr',
                      'samples_byte_nr',
                      'cycles_nr')

FMT_DATA_GROUP_32 = '<2sH4I2H4s'
KEYS_DATA_GROUP_32 = ('id',
                      'block_len',
                      'next_dg_addr',
                      'first_cg_addr',
                      'trigger_addr',
                      'data_block_addr',
                      'cg_nr',
                      'record_id_nr',
                      'reserved0')
FMT_DATA_GROUP = '<2sH4I2H'
KEYS_DATA_GROUP = ('id',
                   'block_len',
                   'next_dg_addr',
                   'first_cg_addr',
                   'trigger_addr',
                   'data_block_addr',
                   'cg_nr',
                   'record_id_nr')

FMT_SOURCE_COMMON = '<2s2H'
FMT_SOURCE_ECU = '<2s3HI80s32s4s'
FMT_SOURCE_EXTRA_ECU = '<HI80s32s4s'
KEYS_SOURCE_ECU = ('id',
                   'block_len',
                   'type',
                   'module_nr',
                   'module_address',
                   'description',
                   'ECU_identification',
                   'reserved0')

FMT_SOURCE_VECTOR = '<2s2H2I36s36s42s'
FMT_SOURCE_EXTRA_VECTOR = '<2I36s36s42s'
KEYS_SOURCE_VECTOR = ('id',
                      'block_len',
                      'type',
                      'CAN_id',
                      'CAN_ch_index',
                      'message_name',
                      'sender_name',
                      'reserved0')

KEYS_TEXT_BLOCK = ('id', 'block_len', 'text')

FMT_CONVERSION_COMMON = FMT_CONVERSION_NONE = '<2s2H2d20s2H'
FMT_CONVERSION_COMMON_SHORT = '<H2d20s2H'
KEYS_CONVESION_NONE = ('id',
                       'block_len',
                       'range_flag',
                       'min_phy_value',
                       'max_phy_value',
                       'unit',
                       'conversion_type',
                       'ref_param_nr')

FMT_CONVERSION_FORMULA = '<2s2H2d20s2H256s'
KEYS_CONVESION_FORMULA = ('id',
                          'block_len',
                          'range_flag',
                          'min_phy_value',
                          'max_phy_value',
                          'unit',
                          'conversion_type',
                          'ref_param_nr',
                          'formula')

FMT_CONVERSION_LINEAR = '<2s2H2d20s2H2d'
KEYS_CONVESION_LINEAR = ('id',
                         'block_len',
                         'range_flag',
                         'min_phy_value',
                         'max_phy_value',
                         'unit',
                         'conversion_type',
                         'ref_param_nr',
                         'b',
                         'a')

FMT_CONVERSION_POLY_RAT = '<2s2H2d20s2H6d'
KEYS_CONVESION_POLY_RAT = ('id',
                           'block_len',
                           'range_flag',
                           'min_phy_value',
                           'max_phy_value',
                           'unit',
                           'conversion_type',
                           'ref_param_nr',
                           'P1',
                           'P2',
                           'P3',
                           'P4',
                           'P5',
                           'P6')

FMT_CONVERSION_EXPO_LOGH = '<2s2H2d20s2H7d'
KEYS_CONVESION_EXPO_LOGH = ('id',
                            'block_len',
                            'range_flag',
                            'min_phy_value',
                            'max_phy_value',
                            'unit',
                            'conversion_type',
                            'ref_param_nr',
                            'P1',
                            'P2',
                            'P3',
                            'P4',
                            'P5',
                            'P6',
                            'P7')

FMT_PROGRAM_BLOCK = '<2sH{}s'
KEYS_PROGRAM_BLOCK = ('id', 'block_len', 'data')

FMT_SAMPLE_REDUCTION_BLOCK = '<2sH3Id'
KEYS_SAMPLE_REDUCTION_BLOCK = ('id',
                               'block_len',
                               'next_sr_addr',
                               'data_block_addr',
                               'cycles_nr',
                               'time_interval')


def get_fmt(data_type, size):
    """"Summary line.

    Extended description of function.

    Parameters
    ----------
    data_type : Type of data_type
        Description of data_type default None
    size : Type of size
        Description of size default None

    Returns
    -------

    Examples
    --------
    >>>>

    """
    if size == 0:
        fmt = 'b'
    if data_type in (DATA_TYPE_UNSIGNED_INTEL, DATA_TYPE_UNSIGNED):
        fmt = '<u{}'.format(size)
    elif data_type == DATA_TYPE_UNSIGNED_MOTOROLA:
        fmt = '>u{}'.format(size)
    elif data_type in (DATA_TYPE_SIGNED_INTEL, DATA_TYPE_SIGNED):
        fmt = '<i{}'.format(size)
    elif data_type == DATA_TYPE_SIGNED_MOTOROLA:
        fmt = '>i{}'.format(size)
    elif data_type in (DATA_TYPE_FLOAT, DATA_TYPE_DOUBLE, DATA_TYPE_FLOAT_INTEL, DATA_TYPE_DOUBLE_INTEL):
        fmt = '<f{}'.format(size)
    elif data_type in (DATA_TYPE_FLOAT_MOTOROLA, DATA_TYPE_DOUBLE_MOTOROLA):
        fmt = '>f{}'.format(size)
    elif data_type == DATA_TYPE_STRING:
        fmt = 'a{}'.format(size)
    return fmt


def fmt_to_datatype(fmt):
    """"Summary line.

    Extended description of function.

    Parameters
    ----------
    fmt : str
        numpy data type string

    Returns
    -------
    data_type, size : int, int
        integer data type as defined by ASAM MDF and byte size

    """
    fmt = str(fmt)
    if 'uint' in fmt:
        size = int(fmt.replace('uint', ''))
        data_type = DATA_TYPE_UNSIGNED
    elif 'int' in fmt:
        size = int(fmt.replace('int', ''))
        data_type = DATA_TYPE_SIGNED
    elif 'float' in fmt:
        size = int(fmt.replace('float', ''))
        data_type = DATA_TYPE_FLOAT if size == 32 else DATA_TYPE_DOUBLE
    elif '|S' in fmt:
        size = int(fmt.replace('|S', ''))
        data_type = DATA_TYPE_STRING
    return data_type, size


def pair(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


class MdfException(Exception):
    pass


class MDF3(object):
    """If the *file_name* exist it will be loaded otherwise an empty file will be created that can be later saved to disk

    Parameters
    ----------
    name : string
        mdf file name
    load_measured_data : bool
        load data option; default *True*

        * if *True* the data group binary data block will be loaded in RAM
        * if *False* the channel data is read from disk on request

    compression : bool
        compression option for data group binary data block; default *False*
    version : string
        mdf file version ('3.00', '3.10' or '3.20'); default '3.20'

    Attributes
    ----------
    name : string
        mdf file name
    groups : list
        list of data groups
    header : OrderedDict
        mdf file header
    file_history : TextBlock
        file history text block; can be None
    load_measured_data : bool
        load measured data option
    compression : bool
        measured data compression option

    """
    def __init__(self, name, load_measured_data=True, compression=False, version='3.20'):
        self.groups = []
        self.header = None
        self.identification = None
        self.file_history = None
        self.name = name
        self.load_measured_data = load_measured_data
        self.compression = compression
        self.channel_db = {}
        self.masters_db = {}

        if os.path.isfile(name):
            self._read()
        else:
            self.groups = []
            self.version = version

            self.identification = FileIdentificationBlock(version=version.encode('ascii'))
            self.header = HeaderBlock(version=int(float(version) * 100))

            self.byteorder = '<'

            self.file_history = TextBlock.from_text('''<FHcomment>
<TX>created</TX>
<tool_id>PythonMDFEditor</tool_id>
<tool_vendor> </tool_vendor>
<tool_version>1.0</tool_version>
</FHcomment>''')

    def append(self, signals, t, names, units, info):
        """
        Appends a new data group.

        Parameters
        ----------
        signals : list
            list on numpy.array signals
        t : numpy.array
            time channel
        names : list
            list of signal name strings
        units : list
            list of signal unit strings
        info : dict
            acquisition information; the 'conversions' key contains a dict of conversions, each key being
            the signal name for which the conversion is specified. See *get* method for the dict description

        Examples
        --------
        >>> # case 1 conversion type None
        >>> s1 = np.array([1, 2, 3, 4, 5])
        >>> s2 = np.array([-1, -2, -3, -4, -5])
        >>> s3 = np.array([0.1, 0.04, 0.09, 0.16, 0.25])
        >>> t = np.array([0.001, 0.002, 0.003, 0.004, 0.005])
        >>> names = ['Positive', 'Negative', 'Float']
        >>> units = ['+', '-', '.f']
        >>> info = {}
        >>> mdf = MDF3('new.mdf')
        >>> mdf.append([s1, s2, s3], t, names, units, info)
        >>> # case 2: VTAB conversions from channels inside another file
        >>> mdf1 = MDF3('in.mdf')
        >>> ch1 = mdf1.get("Channel1_VTAB")
        >>> ch2 = mdf1.get("Channel2_VTABR")
        >>> sigs = [ch1[0], ch2[0]]
        >>> t = ch1[1]
        >>> names = ['NewCh_VTAB', 'NewCh_VTABR']
        >>> units = [ch1[2], ch2[2]]
        >>> info = {'conversions': {'NewCh_VTAB': ch1[3], 'NewCh_VTABR': ch2[3]}
        >>> mdf2 = MDF3('out.mdf')
        >>> mdf2.append(sigs, t, names, units, info)


        """
        self.groups.append({})
        dg_cntr = len(self.groups) - 1
        gp = self.groups[-1]

        channel_nr = len(signals)
        cycles_nr = len(t)

        t_type, t_size = fmt_to_datatype(t.dtype)

        gp['channels'] = gp_channels = []
        gp['channel_conversions'] = gp_conv = []
        gp['channel_extensions'] = gp_source = []
        gp['texts'] = gp_texts = {'channels': [],
                                  'conversion_vtabr': [],
                                  'channel_group': []}

        #time channel texts
        for _, item in gp_texts.items():
            item.append({})

        gp_texts['channel_group'][-1]['comment_addr'] = TextBlock.from_text(
            info.get('acq_comment_addr', 'Python'))

        #channels texts
        for name in names:
            for _, item in gp['texts'].items():
                item.append({})
            if len(name) >= 32:
                gp_texts['channels'][-1]['long_name_addr'] = TextBlock.from_text(name)

        #conversion for time channel
        kargs = {'conversion_type': CONVERSION_TYPE_NONE,
                 'unit': 's'.encode('latin-1'),
                 'min_phy_value': t[0],
                 'max_phy_value': t[-1]}
        gp_conv.append(ChannelConversion(**kargs))

        min_max = [(amin(signal), amax(signal)) for signal in signals]
        #conversion for channels
        for idx, (sig, unit) in enumerate(zip(signals, units)):
            conv = info.get('conversions', {}).get(names[idx], None)
            if conv:
                conv_type = conv['type']
                if conv_type == CONVERSION_TYPE_VTAB:
                    kargs = {}
                    kargs['conversion_type'] = CONVERSION_TYPE_VTAB
                    raw = conv['raw']
                    phys = conv['phys']
                    for i, (r_, p_) in enumerate(zip(raw, phys)):
                        kargs['text_{}'.format(i)] = p_[:31] + b'\x00'
                        kargs['param_val_{}'.format(i)] = r_
                    kargs['ref_param_nr'] = len(raw)
                    kargs['unit'] = unit.encode('latin-1')
                    signals[idx] = sig
                elif conv_type == CONVERSION_TYPE_VTABR:
                    kargs = {}
                    kargs['conversion_type'] = CONVERSION_TYPE_VTABR
                    lower = conv['lower']
                    upper = conv['upper']
                    texts = conv['phys']
                    kargs['unit'] = unit.encode('latin-1')
                    kargs['ref_param_nr'] = len(upper)

                    for i, (u_, l_, t_) in enumerate(zip(upper, lower, texts)):
                        kargs['lower_{}'.format(i)] = l_
                        kargs['upper_{}'.format(i)] = u_
                        kargs['text_{}'.format(i)] = 0
                        gp_texts['conversion_vtabr'][-1]['text_{}'.format(i)] = TextBlock.from_text(t_)

                    signals[idx] = sig
                else:
                     kargs = {'conversion_type': CONVERSION_TYPE_NONE,
                              'unit': units[idx].encode('latin-1') if units[idx] else b'',
                              'min_phy_value': min_max[idx][0],
                              'max_phy_value': min_max[idx][1]}
                gp_conv.append(ChannelConversion(**kargs))
            else:
                kargs = {'conversion_type': CONVERSION_TYPE_NONE,
                         'unit': units[idx].encode('latin-1') if units[idx] else b'',
                         'min_phy_value': min_max[idx][0],
                         'max_phy_value': min_max[idx][1]}
                gp_conv.append(ChannelConversion(**kargs))

        #source for channels and time
        for i in range(channel_nr + 1):
            kargs = {'module_nr': 0,
                     'module_address': 0,
                     'type': SOURCE_ECU,
                     'description': 'Channel inserted by Python Script'.encode('latin-1')}
            gp_source.append(ChannelExtension(**kargs))

        #time channel
        kargs = {'short_name': 't'.encode('latin-1'),
                 #'source_depend_addr': gp['texts']['sources'][0]['path_addr'].address,
                 'channel_type': CHANNEL_TYPE_MASTER,
                 'data_type': t_type,
                 'start_offset': 0,
                 'bit_count': t_size}
        gp_channels.append(Channel(**kargs))
        self.masters_db[dg_cntr] = 0

        sig_dtypes = [sig.dtype for sig in signals]
        sig_formats = [fmt_to_datatype(typ) for typ in sig_dtypes]


        #channels
        offset = t_size
        ch_cntr = 1
        for (sigmin, sigmax), (sig_type, sig_size), name in zip(min_max, sig_formats, names):
            kargs = {'short_name': (name[:31] + '\00').encode('latin-1') if len(name) >= 32 else name.encode('latin-1'),
                     'channel_type': CHANNEL_TYPE_VALUE,
                     'data_type': sig_type,
                     'lower_limit': sigmin,
                     'upper_limit': sigmax,
                     'start_offset': offset,
                     'bit_count': sig_size}
            gp_channels.append(Channel(**kargs))
            offset += sig_size
            self.channel_db[name] = (dg_cntr, ch_cntr)
            ch_cntr += 1

        #channel group
        kargs = {'cycles_nr': cycles_nr,
                 'samples_byte_nr': offset // 8}
        gp['channel_group'] = ChannelGroup(**kargs)
        gp['channel_group']['ch_nr'] = channel_nr + 1

        #data block
        types = [('t', t.dtype),]
        types.extend([('sig{}'.format(i), typ) for i, typ in enumerate(sig_dtypes)])

        arrays = [t, ]
        arrays.extend(signals)

        samples = fromarrays(arrays, dtype=types)
        block = samples.tostring()

        kargs = {'data': block}
        gp['data_block'] = DataBlock(**kargs)

        #data group
        kargs = {'block_len': DG32_BLOCK_SIZE if self.version == 320 else DG31_BLOCK_SIZE}
        gp['data_group'] = DataGroup(**kargs)

    def info(self):
        """get MDF information as a dict

        Examples
        --------
        >>> mdf = MDF3('test.mdf')
        >>> mdf.info()


        """
        info = {}
        info['version'] = self.identification['version_str'].strip(b'\x00').decode('latin-1')
        info['author'] = self.header['author'].strip(b'\x00').decode('latin-1')
        info['organization'] = self.header['organization'].strip(b'\x00').decode('latin-1')
        info['project'] = self.header['project'].strip(b'\x00').decode('latin-1')
        info['subject'] = self.header['subject'].strip(b'\x00').decode('latin-1')
        info['groups'] = len(self.groups)
        for i, gp in enumerate(self.groups):
            inf = {}
            info['group {}'.format(i)] = inf
            inf['cycles'] = gp['channel_group']['cycles_nr']
            inf['channels count'] = len(gp['channels'])
            for j, ch in enumerate(gp['channels']):
                inf['channel {}'.format(j)] = (ch.name, ch['channel_type'])

        return info

    def save(self, dst=None):
        """Save MDF to *dst*. If *dst* is *None* the original file is overwritten

        """
        dst = dst if dst else self.name

        with open(dst, 'wb') as dst:

            #store unique texts and their addresses
            defined_texts = {}
            address = 0

            address += dst.write(bytes(self.identification))
            address += dst.write(bytes(self.header))

            self.file_history.address = address
            address += dst.write(bytes(self.file_history))

            for gp in self.groups:
                gp_texts = gp['texts']

                # Texts
                for _, item_list in gp_texts.items():
                    for my_dict in item_list:
                        for key in my_dict:
                            #text blocks can be shared
                            if my_dict[key].text_str in defined_texts:
                                my_dict[key].address = defined_texts[my_dict[key].text_str]
                            else:
                                defined_texts[my_dict[key].text_str] = address
                                my_dict[key].address = address
                                address += dst.write(bytes(my_dict[key]))

                # ChannelConversions
                cc = gp['channel_conversions']
                for i, conv in enumerate(cc):
                    if conv:
                        conv.address = address
                        if conv['conversion_type'] == CONVERSION_TYPE_VTABR:
                            for key, item in gp_texts['conversion_vtabr'][i].items():
                                conv[key] = item.address

                        address += dst.write(bytes(conv))

                # Channel Extension
                cs = gp['channel_extensions']
                for source in cs:
                    if source:
                        source.address = address
                        address += dst.write(bytes(source))

                # Channels
                # Channels need 4 extra bytes for 8byte alignment

                ch_texts = gp_texts['channels']
                for i, channel in enumerate(gp['channels']):
                    channel.address = address
                    address += CN_BLOCK_SIZE

                    for key in ('long_name_addr', 'comment_addr', 'display_name_addr'):
                        channel_texts = ch_texts[i]
                        if key in channel_texts:
                            channel[key] = channel_texts[key].address
                        else:
                            channel[key] = 0
                    channel['conversion_addr'] = cc[i].address if cc[i] else 0
                    channel['source_depend_addr'] = cs[i].address if cs[i] else 0

                for channel, next_channel in pair(gp['channels']):
                    channel['next_ch_addr'] = next_channel.address
                    dst.write(bytes(channel))
                next_channel['next_ch_addr'] = 0
                dst.write(bytes(next_channel))

                # ChannelGroup
                cg = gp['channel_group']
                cg.address = address

                cg['first_ch_addr'] = gp['channels'][0].address
                cg['next_cg_addr'] = 0
                if 'comment_addr' in gp['texts']['channel_group'][0]:
                    cg['comment_addr'] = gp_texts['channel_group'][0]['comment_addr'].address
                address += dst.write(bytes(cg))


                # DataBlock
                db = gp['data_block']
                db.address = address
                address += dst.write(bytes(db))

            # DataGroup
            for gp in self.groups:

                dg = gp['data_group']
                dg.address = address
                address += dg['block_len']
                dg['first_cg_addr'] = gp['channel_group'].address
                dg['data_block_addr'] = gp['data_block'].address


            for i, dg in enumerate(self.groups[:-1]):
                dg['data_group']['next_dg_addr'] = self.groups[i+1]['data_group'].address
            self.groups[-1]['data_group']['next_dg_addr'] = 0

            for dg in self.groups:
                dst.write(bytes(dg['data_group']))

            if self.groups:
                self.header['first_dg_addr'] = self.groups[0]['data_group'].address
                self.header['dg_nr'] = len(self.groups)
                self.header['comment_addr'] = self.file_history.address
                self.header['program_addr'] = 0
            dst.seek(0, SEEK_START)
            dst.write(bytes(self.identification))
            dst.write(bytes(self.header))

    def get(self, name=None, *, group=None, index=None, raster=None):
        """Gets channel samples.
        Channel can be specified in two ways:

        * using the first positional argument *name*
        * using the group number (keyword argument *group*) and the channel number (keyword argument *index*). Use *info* method for group and channel numbers



        If the *raster* keyword argument is not *None* the output is interpolated accordingly

        Parameters
        ----------
        name : string
            name of channel
        group : int
            0-based group index
        index : int
            0-based channel index
        raster : float
            time raster in seconds

        Returns
        -------
        vals, t, unit, conversion : (numpy.array, numpy.array, string, dict | None)
            The conversion is *None* exept for the VTAB and VTABR conversions. The conversion keys are:

            * for VTAB conversion:

                * raw - numpy.array for X-axis
                * phys - numpy.array of strings for Y-axis
                * type - conversion type = CONVERSION_TYPE_VTAB

            * for VTABR conversion:

                * lower - numpy.array for lower range
                * upper - numpy.array for upper range
                * phys - numpy.array of strings for Y-axis
                * type - conversion type = COONVERSION_TYPE_VTABR

            The conversion information can be used by the *append* method for the *info* argument

        Raises
        ------
        MdfError :

        * if the channel name is not found
        * if the group index is out of range
        * if the channel index is out of range

        """
        if name is None:
            if group is None or index is None:
                raise MdfException('Invalid arguments for "get" methos: must give "name" or, "group" and "index"')
            else:
                gp_nr, ch_nr = group, index
                if gp_nr > len(self.groups) - 1:
                    raise MdfException('Group index out of range')
                if index > len(self.groups[gp_nr]['channels']) - 1:
                    raise MdfException('Channel index out of range')
        else:
            if not name in self.channel_db:
                raise MdfException('Channel "{}" not found'.format(name))
            else:
                gp_nr, ch_nr = self.channel_db[name]

        gp = self.groups[gp_nr]
        channel = gp['channels'][ch_nr]
        conversion = gp['channel_conversions'][ch_nr]
        if conversion:
            unit = conversion['unit'].decode('latin-1').strip('\x00')
        else:
            unit = ''

        time_idx = self.masters_db[gp_nr]
        time_ch = gp['channels'][time_idx]
        time_conv = gp['channel_conversions'][time_idx]

        group = gp

        time_size = time_ch['bit_count'] // 8
        t_fmt = get_fmt(time_ch['data_type'], time_size)
        t_byte_offset, bit_offset = divmod(time_ch['start_offset'], 8)

        bits = time_ch['bit_count']
        if bits % 8:
            t_size = bits // 8 + 1
        else:
            t_size = bits // 8

        bits = channel['bit_count']
        if bits % 8:
            size = bits // 8 + 1
        else:
            size = bits // 8
        block_size = gp['channel_group']['samples_byte_nr'] - gp['data_group']['record_id_nr']
        byte_offset, bit_offset = divmod(channel['start_offset'], 8)
        ch_fmt = get_fmt(channel['data_type'], size)

        if not self.load_measured_data:
            with open(self.name, 'rb') as file_stream:
                # go to the first data block of the current data group
                dat_addr = group['data_group']['data_block_addr']
                read_size = group['channel_group']['samples_byte_nr'] * group['channel_group']['cycles_nr']
                data = DataBlock(file_stream=file_stream, address=dat_addr, size=read_size)['data']
        else:
            data = group['data_block']['data']

        if time_idx == ch_nr:
            types = dtype( [('res1', 'a{}'.format(t_byte_offset)),
                            ('t', t_fmt),
                            ('res2', 'a{}'.format(block_size - byte_offset - size))] )
        if t_byte_offset < byte_offset:
            types = dtype( [('res1', 'a{}'.format(t_byte_offset)),
                            ('t', t_fmt),
                            ('res2', 'a{}'.format(byte_offset - time_size - t_byte_offset)),
                            ('vals', ch_fmt),
                            ('res3', 'a{}'.format(block_size - byte_offset - size))] )
        else:
            types = dtype( [('res1', 'a{}'.format(byte_offset)),
                            ('vals', ch_fmt),
                            ('res2', 'a{}'.format(t_byte_offset - size - byte_offset)),
                            ('t', t_fmt),
                            ('res3', 'a{}'.format(block_size - t_byte_offset - t_size))] )

        values = fromstring(data, types)

        # get timestamps
        time_conv_type = CONVERSION_TYPE_NONE if time_conv is None else time_conv['conversion_type']
        if time_conv_type == CONVERSION_TYPE_LINEAR:
            time_a = time_conv['a']
            time_b = time_conv['b']
            t = values['t'] * time_a
            if time_b:
                t += time_b
        elif time_conv_type == CONVERSION_TYPE_NONE:
            t = values['t']

        if time_idx == ch_nr:
            return t, t, unit, None

        # get channel values
        conversion_type = CONVERSION_TYPE_NONE if conversion is None else conversion['conversion_type']
        vals = values['vals']
        if bit_offset:
            vals.setflags(write=True)
            vals >>= bit_offset
        if bits % 8:
            vals.setflags(write=True)
            vals &= (2**bits - 1)

        if conversion_type == CONVERSION_TYPE_LINEAR:
            a = conversion['a']
            b = conversion['b']
            if (a, b) == (1, 0):
                if not vals.dtype == ch_fmt:
                    vals = vals.astype(ch_fmt)
            else:
                vals = vals * a
                if b:
                    vals.setflags(write=True)
                    vals += b

        elif conversion_type in (CONVERSION_TYPE_TABI, CONVERSION_TYPE_TABX):
            nr = conversion['ref_param_nr']
            raw = array([conversion['raw_{}'.format(i)] for i in range(nr)])
            phys = array([conversion['phys_{}'.format(i)] for i in range(nr)])
            if conversion_type == CONVERSION_TYPE_TABI:
                vals = interp(values['vals'], raw, phys)
            else:
                idx = searchsorted(raw, values['vals'])
                idx = clip(idx, 0, len(raw) - 1)
                vals = phys[idx]

        elif conversion_type == CONVERSION_TYPE_VTAB:
            nr = conversion['ref_param_nr']
            raw = array([conversion['param_val_{}'.format(i)] for i in range(nr)])
            phys = array([conversion['text_{}'.format(i)] for i in range(nr)])
            vals = values['vals']
            info = {'raw': raw, 'phys': phys, 'type': CONVERSION_TYPE_VTAB}

        elif conversion_type == CONVERSION_TYPE_VTABR:
            nr = conversion['ref_param_nr']

            texts = array([gp['texts']['conversion_vtabr'][idx]['text_{}'.format(i)]['text'] for i in range(nr)])
            lower = array([conversion['lower_{}'.format(i)] for i in range(nr)])
            upper = array([conversion['upper_{}'.format(i)] for i in range(nr)])
            vals = values['vals']
            info = {'lower': lower, 'upper': upper, 'phys': texts, 'type': CONVERSION_TYPE_VTABR}

        elif conversion_type in (CONVERSION_TYPE_EXPO, CONVERSION_TYPE_LOGH):
            func = log if conversion_type == CONVERSION_TYPE_EXPO else exp
            P1 = conversion['P1']
            P2 = conversion['P2']
            P3 = conversion['P3']
            P4 = conversion['P4']
            P5 = conversion['P5']
            P6 = conversion['P6']
            P7 = conversion['P7']
            if P4 == 0:
                vals = func(((values['vals'] - P7) * P6 - P3) / P1) / P2
            elif P1 == 0:
                vals = func((P3 / (values['vals'] - P7) - P6) / P4) / P5
            else:
                raise ValueError('wrong conversion type {}'.format(conversion_type))

        elif conversion_type == CONVERSION_TYPE_RAT:
            P1 = conversion['P1']
            P2 = conversion['P2']
            P3 = conversion['P3']
            P4 = conversion['P4']
            P5 = conversion['P5']
            P6 = conversion['P6']
            X = values['vals']
            vals = (P1 * X**2 + P2 * X + P3) / (P4 * X**2 + P5 * X + P6)

        elif conversion_type == CONVERSION_TYPE_POLY:
            P1 = conversion['P1']
            P2 = conversion['P2']
            P3 = conversion['P3']
            P4 = conversion['P4']
            P5 = conversion['P5']
            P6 = conversion['P6']
            X = values['vals']
            vals = (P2 - (P4 * (X - P5 -P6))) / (P3* (X - P5 - P6) - P1)

        elif conversion_type == CONVERSION_TYPE_FORMULA:
            formula = conversion['formula'].decode('latin-1').strip('\x00')
            X1 = values['vals']
            vals = evaluate(formula)

        if raster:
            tx = linspace(0, t[-1], int(t[-1] / raster))
            vals = interp(tx, t, vals)
            t = tx

        if conversion_type in (CONVERSION_TYPE_VTAB, CONVERSION_TYPE_VTABR):
            return vals, t, unit, info
        else:
            return vals, t, unit, None

    def _read(self):
        with open(self.name, 'rb') as file_stream:

            # performance optimization
            read = file_stream.read
            seek = file_stream.seek

            dg_cntr = 0
            seek(0, SEEK_START)

            self.identification = FileIdentificationBlock(file_stream=file_stream)
            self.header = HeaderBlock(file_stream=file_stream)

            self.byteorder = '<' if self.identification['byte_order'] == 0 else '>'

            self.version = self.identification['mdf_version']

            self.file_history = TextBlock(address=self.header['comment_addr'], file_stream=file_stream)

            # go to first date group
            dg_addr = self.header['first_dg_addr']
            # read each data group sequentially
            while dg_addr:
                gp = DataGroup(address=dg_addr, file_stream=file_stream)
                cg_nr = gp['cg_nr']
                cg_addr = gp['first_cg_addr']
                data_addr = gp['data_block_addr']
                new_groups = []
                for i in range(cg_nr):

                    new_groups.append({})
                    grp = new_groups[-1]
                    grp['channels'] = []
                    grp['channel_conversions'] = []
                    grp['channel_extensions'] = []
                    grp['data_block'] = []
                    grp['texts'] = {'channels': [], 'conversion_vtabr': [], 'channel_group': []}

                    kargs = {'first_cg_addr': cg_addr,
                             'data_block_addr': data_addr}
                    if self.version == 320:
                        kargs['block_len'] = DG32_BLOCK_SIZE
                    else:
                        kargs['block_len'] = DG31_BLOCK_SIZE

                    grp['data_group'] = DataGroup(**kargs)

                    # read each channel group sequentially
                    grp['channel_group'] = ChannelGroup(address=cg_addr, file_stream=file_stream)

                    # read acquisition name and comment for current channel group
                    grp['texts']['channel_group'].append({})

                    address = grp['channel_group']['comment_addr']
                    if address:
                        grp['texts']['channel_group'][-1]['comment_addr'] = TextBlock(address=address, file_stream=file_stream)

                    # go to first channel of the current channel group
                    ch_addr = grp['channel_group']['first_ch_addr']
                    ch_cntr = 0
                    grp_chs = grp['channels']
                    grp_conv = grp['channel_conversions']
                    grp_ch_texts = grp['texts']['channels']
                    while ch_addr:
                        # read channel block and create channel object
                        new_ch = Channel(address=ch_addr, file_stream=file_stream)

                        # read conversion block and create channel conversion object
                        address = new_ch['conversion_addr']
                        if address:
                            new_conv = ChannelConversion(address=address, file_stream=file_stream)
                            grp_conv.append(new_conv)
                        else:
                            new_conv = None
                            grp_conv.append(None)

                        vtab_texts = {}
                        if new_conv and new_conv['conversion_type'] == CONVERSION_TYPE_VTABR:
                            for idx in range(new_conv['ref_param_nr']):
                                address = new_conv['text_{}'.format(idx)]
                                if address:
                                    vtab_texts['text_{}'.format(idx)] = TextBlock(address=address, file_stream=file_stream)
                        grp['texts']['conversion_vtabr'].append(vtab_texts)


                        # read source block and create source infromation object
                        address = new_ch['source_depend_addr']
                        if address:
                            grp['channel_extensions'].append(ChannelExtension(address=address, file_stream=file_stream))
                        else:
                            grp['channel_extensions'].append(None)

                        # read text fields for channel
                        ch_texts = {}
                        for key in ('long_name_addr', 'comment_addr', 'display_name_addr'):
                            address = new_ch[key]
                            if address:
                                ch_texts[key] = TextBlock(address=address, file_stream=file_stream)
                        grp_ch_texts.append(ch_texts)

                        # update channel object name and block_size attributes
                        if new_ch['long_name_addr']:
                            new_ch.name = ch_texts['long_name_addr'].text_str
                        else:
                            new_ch.name = new_ch['short_name'].decode('latin-1').strip('\x00')

                        self.channel_db[new_ch.name] = (dg_cntr, ch_cntr)
                        if new_ch['channel_type'] == CHANNEL_TYPE_MASTER:
                            self.masters_db[dg_cntr] = ch_cntr
                        # go to next channel of the current channel group
                        ch_addr = new_ch['next_ch_addr']
                        ch_cntr += 1
                        grp_chs.append(new_ch)

                    cg_addr = grp['channel_group']['next_cg_addr']
                    dg_cntr += 1

                if self.load_measured_data:
                    size = 0
                    record_id_nr = gp['record_id_nr'] if gp['record_id_nr'] <= 2 else 0

                    cg_size = {}
                    cg_data = defaultdict(list)
                    for grp in new_groups:
                        size += (grp['channel_group']['samples_byte_nr'] + record_id_nr) * grp['channel_group']['cycles_nr']
                        cg_size[grp['channel_group']['record_id']] = grp['channel_group']['samples_byte_nr']

                    # read data block of the current data group
                    dat_addr = gp['data_block_addr']
                    seek(dat_addr, SEEK_START)
                    data = read(size)
                    if cg_nr == 1:
                        kargs = {'data': data, 'compression': self.compression}
                        new_groups[0]['data_block'] = DataBlock(**kargs)
                    else:
                        i = 0
                        while i < size:
                            rec_id = data[i]
                            # skip redord id
                            i += 1
                            rec_size = cg_size[rec_id]
                            rec_data = data[i: i+rec_size]
                            cg_data[rec_id].append(rec_data)
                            # if 2 record id's are sued skip also the second one
                            if record_id_nr == 2:
                                i += 1
                            # go to next record
                            i += rec_size
                        for grp in new_groups:
                            kargs = {}
                            kargs['data'] = b''.join(cg_data[grp['channel_group']['record_id']])
                            kargs['compression'] = self.compression
                            grp['channel_group']['record_id'] = 1
                            grp['data_block'] = DataBlock(**kargs)
                    self.groups.extend(new_groups)

                # go to next data group
                dg_addr = gp['next_dg_addr']

    def remove(self, channel_name):
        """
        Remove the channel from the measurement

        Parameters
        ----------
        channel_name : string
            name of the channel to be removed

        Raises
        ------
        NameError : if the channel name is 't'

        """
        if self.load_measured_data:
            if channel_name == 't':
                raise NameError("Can't remove the time channel")
            #print('remove', channel_name, len(self.groups))
            for i, gp in enumerate(self.groups):
                #print(i)
                if channel_name in [ch.name for ch in gp['channels']]:
                    # if this is the only channel in the channel group
                    print('remove', channel_name, i)
                    if len(gp['channels']) == 2:
                        self.groups.pop(i)
                        # if this is the first data group update header information
                        if i == 0:
                            self.header['first_dg_addr'] = self.groups[0]['data_group'].address
                        elif i == len(self.groups) - 1:
                            self.groups[-1]['data_group']['next_dg_addr'] = 0
                        else:
                            self.groups[i-1]['data_group']['next_dg_addr'] = self.groups[i-1]['data_group'].address

                    # else there are other channels in the channel group
                    else:
                        j = [ch.name for ch in gp['channels']].index(channel_name)
                        # remove all text blocks associated with the channel
                        for key in ('channels', 'conversion_vtabr'):
                            gp['texts'][key].pop(j)

                        #print(hex(id(gp)), hex(id(self.groups[i])))

                        channel = gp['channels'].pop(j)
                        start_offset = channel['start_offset']
                        bit_count = channel['bit_count']
                        byte_offset = start_offset // 8
                        byte_size = bit_count // 8

                        # update the other channels informations
                        # especially the addresses and the byte offsets
                        if channel['next_ch_addr'] == 0:
                            gp['channels'][-1]['next_ch_addr'] = 0
                        else:
                            gp['channels'][j-1]['next_ch_addr'] = gp['channels'][j].address

                        # only update byte offset for channels with byte offset higher than the removed channel
                        for channel in gp['channels']:
                            if channel['start_offset'] > start_offset:
                                channel['start_offset'] -= bit_count

                        # update channel group's number of data bytes in each record
                        blocknr = gp['channel_group']['samples_byte_nr'] - gp['data_group']['record_id_nr']
                        gp['channel_group']['samples_byte_nr'] -= byte_size

                        gp['channel_conversions'].pop(j)
                        gp['channel_extensions'].pop(j)

                        gp['channel_group']['ch_nr'] -= 1

                        if byte_size:
                            data = gp['data_block']['data']
                            new_data = bytearray()
                            b_len = len(data)
                            for j in range(b_len // blocknr):
                                position = j * blocknr
                                new_data += data[position: position + byte_offset]
                                new_data += data[position + byte_offset + byte_size: position + blocknr]
                            gp['data_block']['data'] = bytes(new_data)
                    break
            else:
                print('{} not found in {}'.format(channel_name, self.name))
        else:
            print('File opened with the option to not laod the measured data')

    def remove_group(self, *, group_number=None, channel_name=None):
        """
        Removes a data group by providing the group number or a channel name

        Parameters
        ----------
        group_number : int
            data group index
        channel_name : string
            name of a channel inside the data group

        """
        if group_number:
            index = group_number
            if index > len(self.groups):
                raise ValueError('Provided group index is {}, but the measuremetns had only {} groups'.format(
                    index, len(self.groups)))
            index -= 1
        elif channel_name:
            try:
                gp_nr, _ = self.channel_db[channel_name]
            except KeyError:
                raise NameError('Channel "{}" not found in the measurement'.format(channel_name))
            index = gp_nr

        else:
            raise ValueError('No valid group identification provided')

        for channel_name, (dg_nr, cg_nr) in self.channel_db.items():
            if dg_nr == index:
                self.channel_db.pop(channel_name)
        for channel_name, (dg_nr, cg_nr) in self.channel_db.items():
            if dg_nr > index:
                self.channel_db[channel_name] = dg_nr - 1, cg_nr

        self.groups.pop(index)
        if index == 0:
            self.header['first_dg_addr'] = self.groups[0]['data_group'].address
        elif index == len(self.groups) - 1:
            self.groups[-1]['data_group']['next_dg_addr'] = 0
        else:
            self.groups[index - 1]['data_group']['next_dg_addr'] = self.groups[index]['data_group'].address


class Channel(dict):
    ''' CNBLOCK class derived from *dict*

    The Channel object can be created in two modes:

    * using the *file_stream* and *address* keyword parameters - when reading from file
    * using any of the following presented keys - when creating a new Channel

    The keys have the following meaning:

    * id - Block type identifier, always "CN"
    * block_len - Block size of this block in bytes (entire CNBLOCK)
    * next_ch_addr - Pointer to next channel block (CNBLOCK) of this channel group (NIL allowed)
    * conversion_addr - Pointer to the conversion formula (CCBLOCK) of this signal (NIL allowed)
    * source_depend_addr - Pointer to the source-depending extensions (CEBLOCK) of this signal (NIL allowed)
    * ch_depend_addr - Pointer to the dependency block (CDBLOCK) of this signal (NIL allowed)
    * comment_addr - Pointer to the channel comment (TXBLOCK) of this signal (NIL allowed)
    * channel_type - Channel type

        * 0 = data channel
        * 1 = time channel for all signals of this group (in each channel group, exactly one channel must be defined as time channel) The time stamps recording in a time channel are always relative to the start time of the measurement defined in HDBLOCK.

    * short_name - Short signal name, i.e. the first 31 characters of the ASAM-MCD name of the signal (end of text should be indicated by 0)
    * description - Signal description (end of text should be indicated by 0)
    * start_offset - Start offset in bits to determine the first bit of the signal in the data record. The start offset N is divided into two parts: a "Byte offset" (= N div 8) and a "Bit offset" (= N mod 8). The channel block can define an "additional Byte offset" (see below) which must be added to the Byte offset.
    * bit_count - Number of bits used to encode the value of this signal in a data record
    * data_type - Signal data type
    * range_flag - Value range valid flag
    * min_raw_value - Minimum signal value that occurred for this signal (raw value)
    * max_raw_value - Maximum signal value that occurred for this signal (raw value)
    * sampling_rate - Sampling rate for a virtual time channel. Unit [s]
    * long_name_addr - Pointer to TXBLOCK that contains the ASAM-MCD long signal name
    * display_name_addr - Pointer to TXBLOCK that contains the signal's display name (NIL allowed)
    * aditional_byte_offset - Additional Byte offset of the signal in the data record (default value: 0).

    Parameters
    ----------
    file_stream : file handle
        mdf file handle
    address : int
        block address inside mdf file

    Attributes
    ----------
    name : str
        full channel name
    size : int
        size of bytes reprezentation of CNBLOCK
    address : int
        block address inside mdf file

    Examples
    --------
    >>> with open('test.mdf', 'rb') as mdf:
    ...     ch1 = Channel(file_stream=mdf, address=0xBA52)
    >>> ch2 = Channel()
    >>> ch1.name
    'VehicleSpeed'
    >>> ch1['id']
    b'CN'

    '''
    def __init__(self, **kargs):
        super().__init__()

        self.name = ''

        try:
            stream = kargs['file_stream']
            self.address = address = kargs['address']
            stream.seek(address, SEEK_START)
            block = stream.read(CN_BLOCK_SIZE)

            (self['id'],
             self['block_len'],
             self['next_ch_addr'],
             self['conversion_addr'],
             self['source_depend_addr'],
             self['ch_depend_addr'],
             self['comment_addr'],
             self['channel_type'],
             self['short_name'],
             self['description'],
             self['start_offset'],
             self['bit_count'],
             self['data_type'],
             self['range_flag'],
             self['min_raw_value'],
             self['max_raw_value'],
             self['sampling_rate'],
             self['long_name_addr'],
             self['display_name_addr'],
             self['aditional_byte_offset']) = unpack(FMT_CHANNEL, block)

        except KeyError:

            self.address = 0
            self['id'] = kargs.get('id', 'CN'.encode('latin-1'))
            self['block_len'] = kargs.get('block_len', CN_BLOCK_SIZE)
            self['next_ch_addr'] = kargs.get('next_ch_addr', 0)
            self['conversion_addr'] = kargs.get('conversion_addr', 0)
            self['source_depend_addr'] = kargs.get('source_depend_addr', 0)
            self['ch_depend_addr'] = kargs.get('ch_depend_addr', 0)
            self['comment_addr'] = kargs.get('comment_addr', 0)
            self['channel_type'] = kargs.get('channel_type', 0)
            self['short_name'] = kargs.get('short_name', ('\x00'*32).encode('latin-1'))
            self['description'] = kargs.get('description', ('\x00'*32).encode('latin-1'))
            self['start_offset'] = kargs.get('start_offset', 0)
            self['bit_count'] = kargs.get('bit_count', 8)
            self['data_type'] = kargs.get('data_type', 0)
            self['range_flag'] = kargs.get('range_flag', 1)
            self['min_raw_value'] = kargs.get('min_raw_value', 0)
            self['max_raw_value'] = kargs.get('max_raw_value', 0)
            self['sampling_rate'] = kargs.get('sampling_rate', 0)
            self['long_name_addr'] = kargs.get('long_name_addr', 0)
            self['display_name_addr'] = kargs.get('display_name_addr', 0)
            self['aditional_byte_offset'] = kargs.get('aditional_byte_offset', 0)

    def __bytes__(self):
        return pack(FMT_CHANNEL, *[self[key] for key in KEYS_CHANNEL])


class ChannelConversion(dict):
    ''' CCBLOCK class derived from *dict*

    The ChannelConversion object can be created in two modes:

    * using the *file_stream* and *address* keyword parameters - when reading from file
    * using any of the following presented keys - when creating a new ChannelConversion

    The first keys are common for all conversion types, and are followed by conversion specific keys. The keys have the following meaning:

    * common keys

        * id - Block type identifier, always "CC"
        * block_len - Block size of this block in bytes (entire CCBLOCK)
        * range_flag - Physical value range valid flag:
        * min_phy_value - Minimum physical signal value that occurred for this signal
        * max_phy_value - Maximum physical signal value that occurred for this signal
        * unit - Physical unit (string should be terminated with 0)
        * conversion_type - Conversion type (formula identifier)
        * ref_param_nr - Size information about additional conversion data

    * specific keys

        * linear conversion

            * b - offset
            * a - factor
            * CANapeHiddenExtra - sometimes CANape appends extra information; not compliant with MDF specs

        * ASAM formula conversion

            * formula - ecuation as string

        * polynomial or rational conversion

            * P1 .. P6 - factors

        * exponential or logarithmic conversion

            * P1 .. P7 - factors

        * tabular with or without interpolation (grouped by *n*)

            * raw_{n} - n-th raw integer value (X axis)
            * phys_{n} - n-th physical value (Y axis)

        * text table conversion

            * param_val_{n} - n-th integers value (X axis)
            * text_{n} - n-th text value (Y axis)

        * text range table conversion

            * lower_{n} - n-th lower raw value
            * upper_{n} - n-th upper raw value
            * text_{n} - n-th text value

    Parameters
    ----------
    file_stream : file handle
        mdf file handle
    address : int
        block address inside mdf file

    Attributes
    ----------
    size : int
        size of bytes reprezentation of CCBLOCK
    address : int
        block address inside mdf file

    Examples
    --------
    >>> with open('test.mdf', 'rb') as mdf:
    ...     cc1 = ChannelConversion(file_stream=mdf, address=0xBA52)
    >>> cc2 = ChannelConversion(conversion_type=0)
    >>> cc1['b'], cc1['a']
    0, 100.0

    '''
    def __init__(self, **kargs):
        super().__init__()

        try:
            stream = kargs['file_stream']
            self.address = address = kargs['address']
            stream.seek(address, SEEK_START)
            block = stream.read(4)
            (self['id'],
             self['block_len']) = unpack('<2sH', block)
            size = self['block_len']
            block += stream.read(size - 4)

            (self['range_flag'],
             self['min_phy_value'],
             self['max_phy_value'],
             self['unit'],
             self['conversion_type'],
             self['ref_param_nr']) = unpack(FMT_CONVERSION_COMMON_SHORT, block[4:CC_COMMON_BLOCK_SIZE])

            conv_type = self['conversion_type']

            if conv_type == CONVERSION_TYPE_NONE:
                pass
            elif conv_type == CONVERSION_TYPE_FORMULA:
                self['formula'] = unpack('<{}s'.format(size - 46), block[CC_COMMON_BLOCK_SIZE:])[0]

            elif conv_type in (CONVERSION_TYPE_TABI, CONVERSION_TYPE_TABX):
                for i, (raw, phys) in enumerate(iter_unpack('<2d', block[CC_COMMON_BLOCK_SIZE:])):
                    (self['raw_{}'.format(i)],
                     self['phys_{}'.format(i)]) = raw, phys

            elif conv_type == CONVERSION_TYPE_LINEAR:
                (self['b'],
                 self['a']) = unpack('<2d', block[CC_COMMON_BLOCK_SIZE: CC_LIN_BLOCK_SIZE])
                if not size == CC_LIN_BLOCK_SIZE:
                    self['CANapeHiddenExtra'] = block[CC_LIN_BLOCK_SIZE:]

            elif conv_type in (CONVERSION_TYPE_POLY, CONVERSION_TYPE_RAT):
                (self['P1'],
                 self['P2'],
                 self['P3'],
                 self['P4'],
                 self['P5'],
                 self['P6']) = unpack('<6d', block[CC_COMMON_BLOCK_SIZE: CC_POLY_BLOCK_SIZE])

            elif conv_type in (CONVERSION_TYPE_EXPO, CONVERSION_TYPE_LOGH):
                (self['P1'],
                 self['P2'],
                 self['P3'],
                 self['P4'],
                 self['P5'],
                 self['P6'],
                 self['P7']) = unpack('<7d', block[CC_COMMON_BLOCK_SIZE: CC_EXPO_BLOCK_SIZE])

            elif conv_type == CONVERSION_TYPE_VTAB:
                nr = self['ref_param_nr']

                for i, (val, text) in enumerate(iter_unpack('<d32s', block[CC_COMMON_BLOCK_SIZE:])):
                    (self['param_val_{}'.format(i)],
                     self['text_{}'.format(i)]) = val, text

            elif conv_type == CONVERSION_TYPE_VTABR:
                nr = self['ref_param_nr']

                for i, (lower, upper, text) in enumerate(iter_unpack('<2dI', block[CC_COMMON_BLOCK_SIZE:])):
                    (self['lower_{}'.format(i)],
                     self['upper_{}'.format(i)],
                     self['text_{}'.format(i)]) = lower, upper, text
        except KeyError:

            self.address = 0
            self['id'] = 'CC'.encode('latin-1')

            if kargs['conversion_type'] == CONVERSION_TYPE_NONE:
                self['block_len'] = kargs.get('block_len', CC_COMMON_BLOCK_SIZE)
                self['range_flag'] = kargs.get('range_flag', 1)
                self['min_phy_value'] = kargs.get('min_phy_value', 0)
                self['max_phy_value'] = kargs.get('max_phy_value', 0)
                self['unit'] = kargs.get('unit', ('\x00'*20).encode('latin-1'))
                self['conversion_type'] = CONVERSION_TYPE_NONE
                self['ref_param_nr'] = kargs.get('ref_param_nr', 0)

            elif kargs['conversion_type'] == CONVERSION_TYPE_LINEAR:
                self['block_len'] = kargs.get('block_len', CC_LIN_BLOCK_SIZE)
                self['range_flag'] = kargs.get('range_flag', 1)
                self['min_phy_value'] = kargs.get('min_phy_value', 0)
                self['max_phy_value'] = kargs.get('max_phy_value', 0)
                self['unit'] = kargs.get('unit', ('\x00'*20).encode('latin-1'))
                self['conversion_type'] = CONVERSION_TYPE_LINEAR
                self['ref_param_nr'] = kargs.get('ref_param_nr', 2)
                self['b'] = kargs.get('b', 0)
                self['a'] = kargs.get('a', 1)
                if not self['block_len'] == CC_LIN_BLOCK_SIZE:
                    self['CANapeHiddenExtra'] = kargs['CANapeHiddenExtra']

            elif kargs['conversion_type'] in (CONVERSION_TYPE_POLY, CONVERSION_TYPE_RAT):
                self['block_len'] = kargs.get('block_len', CC_POLY_BLOCK_SIZE)
                self['range_flag'] = kargs.get('range_flag', 1)
                self['min_phy_value'] = kargs.get('min_phy_value', 0)
                self['max_phy_value'] = kargs.get('max_phy_value', 0)
                self['unit'] = kargs.get('unit', ('\x00'*20).encode('latin-1'))
                self['conversion_type'] = kargs.get('conversion_type', CONVERSION_TYPE_POLY)
                self['ref_param_nr'] = kargs.get('ref_param_nr', 2)
                self['P1'] = kargs.get('P1', 0)
                self['P2'] = kargs.get('P2', 0)
                self['P3'] = kargs.get('P3', 0)
                self['P4'] = kargs.get('P4', 0)
                self['P5'] = kargs.get('P5', 0)
                self['P6'] = kargs.get('P6', 0)

            elif kargs['conversion_type'] in (CONVERSION_TYPE_EXPO, CONVERSION_TYPE_LOGH):
                self['block_len'] = kargs.get('block_len', CC_EXPO_BLOCK_SIZE)
                self['range_flag'] = kargs.get('range_flag', 1)
                self['min_phy_value'] = kargs.get('min_phy_value', 0)
                self['max_phy_value'] = kargs.get('max_phy_value', 0)
                self['unit'] = kargs.get('unit', ('\x00'*20).encode('latin-1'))
                self['conversion_type'] = kargs.get('conversion_type', CONVERSION_TYPE_EXPO)
                self['ref_param_nr'] = kargs.get('ref_param_nr', 2)
                self['P1'] = kargs.get('P1', 0)
                self['P2'] = kargs.get('P2', 0)
                self['P3'] = kargs.get('P3', 0)
                self['P4'] = kargs.get('P4', 0)
                self['P5'] = kargs.get('P5', 0)
                self['P6'] = kargs.get('P6', 0)
                self['P7'] = kargs.get('P7', 0)

            elif kargs['conversion_type'] == CONVERSION_TYPE_FORMULA:
                self['block_len'] = kargs.get('block_len', CC_POLY_BLOCK_SIZE)
                self['range_flag'] = kargs.get('range_flag', 1)
                self['min_phy_value'] = kargs.get('min_phy_value', 0)
                self['max_phy_value'] = kargs.get('max_phy_value', 0)
                self['unit'] = kargs.get('unit', ('\x00'*20).encode('latin-1'))
                self['conversion_type'] = kargs.get('conversion_type', CONVERSION_TYPE_FORMULA)
                self['ref_param_nr'] = kargs.get('ref_param_nr', 2)
                self['formula'] = kargs.get('formula', b'X1'+b'\x00'*254)
            elif kargs['conversion_type'] in (CONVERSION_TYPE_TABI, CONVERSION_TYPE_TABX):
                nr = kargs['ref_param_nr']
                self['block_len'] = kargs['block_len']
                self['range_flag'] = kargs.get('range_flag', 1)
                self['min_phy_value'] = kargs.get('min_phy_value', 0)
                self['max_phy_value'] = kargs.get('max_phy_value', 0)
                self['unit'] = kargs.get('unit', ('\x00'*20).encode('latin-1'))
                self['conversion_type'] = kargs.get('conversion_type', CONVERSION_TYPE_TABI)
                self['ref_param_nr'] = kargs.get('ref_param_nr', 2)
                for i in range(nr):
                    self['raw_{}'.format(i)] = kargs['raw_{}'.format(i)]
                    self['phys_{}'.format(i)] = kargs['phys_{}'.format(i)]

            elif kargs['conversion_type'] == CONVERSION_TYPE_VTAB:
                nr = kargs['ref_param_nr']
                self['block_len'] = kargs.get('block_len', CC_COMMON_BLOCK_SIZE + 40*nr)
                self['range_flag'] = kargs.get('range_flag', 0)
                self['min_phy_value'] = kargs.get('min_phy_value', 0)
                self['max_phy_value'] = kargs.get('max_phy_value', 0)
                self['unit'] = kargs.get('unit', ('\x00'*20).encode('latin-1'))
                self['conversion_type'] = CONVERSION_TYPE_VTAB
                self['ref_param_nr'] = nr

                for i in range(nr):
                    self['param_val_{}'.format(i)] = kargs['param_val_{}'.format(i)]
                    self['text_{}'.format(i)] = kargs['text_{}'.format(i)]

            elif kargs['conversion_type'] == CONVERSION_TYPE_VTABR:
                nr = kargs.get('ref_param_nr', 0)
                self['block_len'] = kargs.get('block_len', CC_COMMON_BLOCK_SIZE + 20*nr)
                self['range_flag'] = kargs.get('range_flag', 0)
                self['min_phy_value'] = kargs.get('min_phy_value', 0)
                self['max_phy_value'] = kargs.get('max_phy_value', 0)
                self['unit'] = kargs.get('unit', ('\x00'*20).encode('latin-1'))
                self['conversion_type'] = CONVERSION_TYPE_VTABR
                self['ref_param_nr'] = kargs.get('ref_param_nr', 0)

                for i in range(self['ref_param_nr']):
                    self['lower_{}'.format(i)] = kargs['lower_{}'.format(i)]
                    self['upper_{}'.format(i)] = kargs['upper_{}'.format(i)]
                    self['text_{}'.format(i)] = kargs['text_{}'.format(i)]
            else:
                raise Exception('Conversion type "{}" not implemented'.format(kargs['conversion_type']))

    def __bytes__(self):
        conv = self['conversion_type']
        if conv == CONVERSION_TYPE_NONE:
            fmt = FMT_CONVERSION_COMMON
            keys = KEYS_CONVESION_NONE
        elif conv == CONVERSION_TYPE_FORMULA:
            fmt = FMT_CONVERSION_FORMULA
            keys = KEYS_CONVESION_FORMULA
        elif conv == CONVERSION_TYPE_LINEAR:
            fmt = FMT_CONVERSION_LINEAR
            keys = KEYS_CONVESION_LINEAR
            if not self['block_len'] == CC_LIN_BLOCK_SIZE:
                fmt += '{}s'.format(self['block_len'] - CC_LIN_BLOCK_SIZE)
                keys += ('CANapeHiddenExtra',)
        elif conv in (CONVERSION_TYPE_POLY, CONVERSION_TYPE_RAT):
            fmt = FMT_CONVERSION_POLY_RAT
            keys = KEYS_CONVESION_POLY_RAT
        elif conv in (CONVERSION_TYPE_EXPO, CONVERSION_TYPE_LOGH):
            fmt = FMT_CONVERSION_EXPO_LOGH
            keys = KEYS_CONVESION_EXPO_LOGH
        elif conv in (CONVERSION_TYPE_TABI, CONVERSION_TYPE_TABX):
            nr = self['ref_param_nr']
            fmt = FMT_CONVERSION_COMMON + '{}d'.format(nr * 2)
            keys = list(KEYS_CONVESION_NONE)
            for i in range(nr):
                keys.append('raw_{}'.format(i))
                keys.append('phys_{}'.format(i))
        elif conv == CONVERSION_TYPE_VTABR:
            nr = self['ref_param_nr']
            fmt = FMT_CONVERSION_COMMON + '2dI' * nr
            keys = list(KEYS_CONVESION_NONE)
            for i in range(nr):
                keys.append('lower_{}'.format(i))
                keys.append('upper_{}'.format(i))
                keys.append('text_{}'.format(i))
        elif conv == CONVERSION_TYPE_VTAB:
            nr = self['ref_param_nr']
            fmt = FMT_CONVERSION_COMMON + 'd32s' * nr
            keys = list(KEYS_CONVESION_NONE)
            for i in range(nr):
                keys.append('param_val_{}'.format(i))
                keys.append('text_{}'.format(i))

        return pack(fmt, *[self[key] for key in keys])


class ChannelDependency(dict):
    ''' CDBLOCK class derived from *dict*

    Currently the ChannelDependency object can only be created using the *file_stream* and *address* keyword parameters when reading from file

    The keys have the following meaning:

    * id - Block type identifier, always "CD"
    * block_len - Block size of this block in bytes (entire CDBLOCK)
    * data - Dependency type
    * sd_nr - Total number of signals dependencies (m)
    * for each dependency there is a group of three keys:

        * dg_{n} - Pointer to the data group block (DGBLOCK) of signal dependency *n*
        * cg_{n} - Pointer to the channel group block (DGBLOCK) of signal dependency *n*
        * ch_{n} - Pointer to the channel block (DGBLOCK) of signal dependency *n*

    * there can also be optional keys which decribe dimensions for the N-dimensional dependencies:

        * dim_{n} - Optional: size of dimension *n* for N-dimensional dependency

    Parameters
    ----------
    file_stream : file handle
        mdf file handle
    address : int
        block address inside mdf file

    Attributes
    ----------
    size : int
        size of bytes reprezentation of PRBLOCK
    address : int
        block address inside mdf file

    '''
    def __init__(self, **kargs):
        super().__init__()

        try:
            stream = kargs['file_stream']
            self.address = address = kargs['address']
            stream.seek(address, SEEK_START)

            (self['id'],
             self['block_len'],
             self['dependency_type'],
             self['sd_nr']) = unpack('<2s3H', stream.read(50))

            links_size = 3 * 4 * self['sd_nr']
            links = unpack('<{}I'.format(3 * self['sd_nr']), stream.read(links_size))

            for i in range(self['sd_nr']):
                self['dg_{}'.format(i)] = links[i]
                self['cg_{}'.format(i)] = links[i+1]
                self['ch_{}'.format(i)] = links[i+2]

            optional_dims_nr = (self['block_len'] - 50 - links_size) // 2
            self['optional_dims_nr'] = optional_dims_nr
            if optional_dims_nr:
                dims = unpack('<{}H'.format(optional_dims_nr), stream.read(optional_dims_nr * 2))
                for i, dim in enumerate(dims):
                    self['dim_{}'.format(i)] = dim

        except KeyError:
            print('CDBLOCK can only be loaded from a mdf file')

    def __bytes__(self):
        fmt = '<2s3H{}I'.format(self['sd_nr'] * 3)
        keys = ('id', 'block_len', 'dependency_type', 'sd_nr')
        for i in range(self['sd_nr']):
            keys += ('dg_{}'.format(i), 'cg_{}'.format(i), 'ch_{}'.format(i))
        if self['optional_dims_nr']:
            fmt += '{}H'.format(self['optional_dims_nr'])
            keys += tuple('dim_{}'.format(i) for i in range(self['optional_dims_nr']))
        return pack(fmt, *[self[key] for key in keys])


class ChannelExtension(dict):
    ''' CEBLOCK class derived from *dict*

    The ChannelExtension object can be created in two modes:

    * using the *file_stream* and *address* keyword parameters - when reading from file
    * using any of the following presented keys - when creating a new ChannelExtension

    The first keys are common for all conversion types, and are followed by conversion specific keys. The keys have the following meaning:

    * common keys

        * id - Block type identifier, always "CE"
        * block_len - Block size of this block in bytes (entire CEBLOCK)
        * type - Extension type identifier

    * specific keys

        * for DIM block

            * module_nr - Number of module
            * module_address - Address
            * description - Description
            * ECU_identification - Identification of ECU
            * reserved0' - reserved

        * for Vector CAN block

            * CAN_id - Identifier of CAN message
            * CAN_ch_index - Index of CAN channel
            * message_name - Name of message (string should be terminated by 0)
            * sender_name - Name of sender (string should be terminated by 0)
            * reserved0 - reserved

    Parameters
    ----------
    file_stream : file handle
        mdf file handle
    address : int
        block address inside mdf file

    Attributes
    ----------
    size : int
        size of bytes reprezentation of CEBLOCK
    address : int
        block address inside mdf file

    '''
    def __init__(self, **kargs):
        super().__init__()

        try:
            stream = kargs['file_stream']
            self.address = address = kargs['address']
            stream.seek(address, SEEK_START)
            (self['id'],
             self['block_len'],
             self['type']) = unpack(FMT_SOURCE_COMMON, stream.read(6))
            block = stream.read(self['block_len'] - 6)

            if self['type'] == SOURCE_ECU:
                (self['module_nr'],
                 self['module_address'],
                 self['description'],
                 self['ECU_identification'],
                 self['reserved0']) = unpack(FMT_SOURCE_EXTRA_ECU, block)
            elif self['type'] == SOURCE_VECTOR:
                (self['CAN_id'],
                 self['CAN_ch_index'],
                 self['message_name'],
                 self['sender_name'],
                 self['reserved0']) = unpack(FMT_SOURCE_EXTRA_VECTOR, block)
        except KeyError:

            self.address = 0
            self['id'] = kargs.get('id', 'CE'.encode('latin-1'))
            self['block_len'] = kargs.get('block_len', CE_BLOCK_SIZE)
            self['type'] = kargs.get('type', 2)
            if self['type'] == SOURCE_ECU:
                self['module_nr'] = kargs.get('module_nr', 0)
                self['module_address'] = kargs.get('module_address', 0)
                self['description'] = kargs.get('description', '\x00'.encode('latin-1'))
                self['ECU_identification'] = kargs.get('ECU_identification', '\x00'.encode('latin-1'))
                self['reserved0'] = kargs.get('reserved0', '\x00'.encode('latin-1'))
            elif self['type'] == SOURCE_VECTOR:
                self['CAN_id'] = kargs.get('CAN_id', 0)
                self['CAN_ch_index'] = kargs.get('CAN_ch_index', 0)
                self['message_name'] = kargs.get('message_name', '\x00'.encode('latin-1'))
                self['sender_name'] = kargs.get('sender_name', '\x00'.encode('latin-1'))
                self['reserved0'] = kargs.get('reserved0', '\x00'.encode('latin-1'))

    def __bytes__(self):
        typ = self['type']
        if typ == SOURCE_ECU:
            fmt = FMT_SOURCE_ECU
            keys = KEYS_SOURCE_ECU
        else:
            fmt = FMT_SOURCE_VECTOR
            keys = KEYS_SOURCE_VECTOR
        return pack(fmt, *[self[key] for key in keys])


class ChannelGroup(dict):
    ''' CGBLOCK class derived from *dict*

    The ChannelGroup object can be created in two modes:

    * using the *file_stream* and *address* keyword parameters - when reading from file
    * using any of the following presented keys - when creating a new ChannelGroup

    The keys have the following meaning:

    * id - Block type identifier, always "CG"
    * block_len - Block size of this block in bytes (entire CGBLOCK)
    * next_cg_addr - Pointer to next channel group block (CGBLOCK) (NIL allowed)
    * first_ch_addr - Pointer to first channel block (CNBLOCK) (NIL allowed)
    * comment_addr - Pointer to channel group comment text (TXBLOCK) (NIL allowed)
    * record_id - Record ID, i.e. value of the identifier for a record if the DGBLOCK defines a number of record IDs > 0
    * ch_nr - Number of channels (redundant information)
    * samples_byte_nr - Size of data record in Bytes (without record ID), i.e. size of plain data for a each recorded sample of this channel group
    * cycles_nr - Number of records of this type in the data block i.e. number of samples for this channel group
    * sample_reduction_addr - only since version 3.3. Pointer to first sample reduction block (SRBLOCK) (NIL allowed) Default value: NIL.

    Parameters
    ----------
    file_stream : file handle
        mdf file handle
    address : int
        block address inside mdf file

    Attributes
    ----------
    size : int
        size of bytes reprezentation of CGBLOCK
    address : int
        block address inside mdf file

    Examples
    --------
    >>> with open('test.mdf', 'rb') as mdf:
    ...     cg1 = ChannelGroup(file_stream=mdf, address=0xBA52)
    >>> cg2 = ChannelGroup(sample_bytes_nr=32)
    >>> hex(cg1.address)
    0xBA52
    >>> cg1['id']
    b'CG'

    '''
    def __init__(self, **kargs):
        super().__init__()

        try:
            stream = kargs['file_stream']
            self.address = address = kargs['address']
            stream.seek(address, SEEK_START)
            block = stream.read(CG_BLOCK_SIZE)

            (self['id'],
             self['block_len'],
             self['next_cg_addr'],
             self['first_ch_addr'],
             self['comment_addr'],
             self['record_id'],
             self['ch_nr'],
             self['samples_byte_nr'],
             self['cycles_nr']) = unpack(FMT_CHANNEL_GROUP, block)
            if self['block_len'] == CG33_BLOCK_SIZE:
                self['sample_reduction_addr'] = unpack('<I', stream.read(4))[0]
        except KeyError:
            self.address = 0
            self['id'] = kargs.get('id', 'CG'.encode('latin-1'))
            self['block_len'] = kargs.get('block_len', CG_BLOCK_SIZE)
            self['next_cg_addr'] = kargs.get('next_cg_addr', 0)
            self['first_ch_addr'] = kargs.get('first_ch_addr', 0)
            self['comment_addr'] = kargs.get('comment_addr', 0)
            self['record_id'] = kargs.get('record_id', 1)
            self['ch_nr'] = kargs.get('ch_nr', 0)
            self['samples_byte_nr'] = kargs.get('samples_byte_nr', 0)
            self['cycles_nr'] = kargs.get('cycles_nr', 0)
            if self['block_len'] == CG33_BLOCK_SIZE:
                self['sample_reduction_addr'] = 0

    def __bytes__(self):
        fmt = FMT_CHANNEL_GROUP
        keys = KEYS_CHANNEL_GROUP
        if self['block_len'] == CG33_BLOCK_SIZE:
            fmt += 'I'
            keys += ('sample_reduction_addr',)

        return pack(fmt, *[self[key] for key in keys])


class DataBlock(dict):
    """Data Block class derived from *dict*

    Data can be compressed to lower RAM usage if the *compression* keyword if set to True.

    The DataBlock object can be created in two modes:

    * using the *file_stream*, *address* and *size* keyword parameters - when reading from file
    * using any of the following presented keys - when creating a new ChannelGroup

    The keys have the following meaning:

    * data - bytes block

    Attributes
    ----------
    size : int
        length of uncompressed samples
    address : int
        block address
    compression : bool
        compression flag

    Parameters
    ----------
    address : int
        block address inside the measurement file
    stream : file.io.handle
        binary file stream
    size : int
        block size
    compression : bool
        option flag for data compression; default *False*

    """

    def __init__(self, **kargs):
        super().__init__()

        try:
            stream = kargs['file_stream']
            size = kargs['size']
            self.address = address = kargs['address']
            stream.seek(address, SEEK_START)

            self.compression = kargs.get('compression', False)
            self['data'] = stream.read(size)


        except KeyError:
            self.address = 0
            self.compression = kargs.get('compression', False)
            self['data'] = kargs.get('data', b'')

    def __setitem__(self, item, value):
        if item == 'data':
            if self.compression:
                super().__setitem__(item, compress(value, 8))
            else:
                super().__setitem__(item, value)
        else:
            super().__setitem__(item, value)

    def __getitem__(self, item):
        if item == 'data' and self.compression:
            return decompress(super().__getitem__(item))
        else:
            return super().__getitem__(item)

    def __bytes__(self):
        return self['data']


class DataGroup(dict):
    ''' DGBLOCK class derived from *dict*

    The DataGroup object can be created in two modes:

    * using the *file_stream* and *address* keyword parameters - when reading from file
    * using any of the following presented keys - when creating a new DataGroup

    The keys have the following meaning:

    * id - Block type identifier, always "DG"
    * block_len - Block size of this block in bytes (entire DGBLOCK)
    * next_dg_addr - Pointer to next data group block (DGBLOCK) (NIL allowed)
    * first_cg_addr - Pointer to first channel group block (CGBLOCK) (NIL allowed)
    * trigger_addr - Pointer to trigger block (TRBLOCK) (NIL allowed)
    * data_block_addr - Pointer to the data block (see separate chapter on data storage)
    * cg_nr - Number of channel groups (redundant information)
    * record_id_nr - Number of record IDs in the data block
    * reserved0 - since version 3.2; Reserved

    Parameters
    ----------
    file_stream : file handle
        mdf file handle
    address : int
        block address inside mdf file

    Attributes
    ----------
    size : int
        size of bytes reprezentation of DGBLOCK
    address : int
        block address inside mdf file

    '''
    def __init__(self, **kargs):
        super().__init__()

        try:
            stream = kargs['file_stream']
            self.address = address = kargs['address']
            stream.seek(address, SEEK_START)
            block = stream.read(DG31_BLOCK_SIZE)

            (self['id'],
             self['block_len'],
             self['next_dg_addr'],
             self['first_cg_addr'],
             self['trigger_addr'],
             self['data_block_addr'],
             self['cg_nr'],
             self['record_id_nr']) = unpack(FMT_DATA_GROUP, block)

            if self['block_len'] == DG32_BLOCK_SIZE:
                self['reserved0'] = stream.read(4)
        except KeyError:
            self.address = 0
            self['id'] = kargs.get('id', 'DG'.encode('latin-1'))
            self['block_len'] = kargs.get('block_len', DG32_BLOCK_SIZE)
            self['next_dg_addr'] = kargs.get('next_dg_addr', 0)
            self['first_cg_addr'] = kargs.get('first_cg_addr', 0)
            self['trigger_addr'] = kargs.get('comment_addr', 0)
            self['data_block_addr'] = kargs.get('data_block_addr', 0)
            self['cg_nr'] = kargs.get('cg_nr', 1)
            self['record_id_nr'] = kargs.get('record_id_nr', 0)
            if self['block_len'] == DG32_BLOCK_SIZE:
                self['reserved0'] = b'\x00\x00\x00\x00'

    def __bytes__(self):
        if self['block_len'] == DG32_BLOCK_SIZE:
            fmt = FMT_DATA_GROUP_32
            keys = KEYS_DATA_GROUP_32
        else:
            fmt = FMT_DATA_GROUP
            keys = KEYS_DATA_GROUP
        return pack(fmt, *[self[key] for key in keys])


class FileIdentificationBlock(dict):
    ''' IDBLOCK class derived from *dict*

    The TriggerBlock object can be created in two modes:

    * using the *file_stream* and *address* keyword parameters - when reading from file
    * using the classmethod *from_text*

    The keys have the following meaning:

    * file_identification -  file identifier
    * version_str - format identifier
    * program_identification - program identifier
    * byte_order - default byte order
    * float_format - default floating-point format
    * mdf_version - version number of MDF format
    * code_page - code page number
    * reserved0 - reserved
    * reserved1 - reserved
    * unfinalized_standard_flags - Standard Flags for unfinalized MDF
    * unfinalized_custom_flags - Custom Flags for unfinalized MDF

    Parameters
    ----------
    file_stream : file handle
        mdf file handle

    version : int
        mdf version in case of new file

    '''
    def __init__(self, **kargs):
        super().__init__()

        self.address = 0
        try:

            stream = kargs['file_stream']
            stream.seek(0, SEEK_START)

            (self['file_identification'],
             self['version_str'],
             self['program_identification'],
             self['byte_order'],
             self['float_format'],
             self['mdf_version'],
             self['code_page'],
             self['reserved0'],
             self['reserved1'],
             self['unfinalized_standard_flags'],
             self['unfinalized_custom_flags']) = unpack(ID_FMT, stream.read(ID_BLOCK_SIZE))
        except KeyError:
            self['file_identification'] = 'MDF     '.encode('latin-1')
            self['version_str'] = kargs.get('version', b'330' + b'\x00' * 5)
            self['program_identification'] = 'Python  '.encode('latin-1')
            self['byte_order'] = BYTE_ORDER_INTEL
            self['float_format'] = 0
            self['mdf_version'] = int(float(self['version_str']) * 100)
            self['code_page'] = 0
            self['reserved0'] = b'\x00' * 2
            self['reserved1'] = b'\x00' * 26
            self['unfinalized_standard_flags'] = 0
            self['unfinalized_custom_flags'] = 0

    def __bytes__(self):
        return pack(ID_FMT, *[self[key] for key in ID_KEYS])


class HeaderBlock(dict):
    ''' HDBLOCK class derived from *dict*

    The TriggerBlock object can be created in two modes:

    * using the *file_stream* - when reading from file
    * using the classmethod *from_text*

    The keys have the following meaning:

    * id - Block type identifier, always "HD"
    * block_len - Block size of this block in bytes (entire HDBLOCK)
    * first_dg_addr - Pointer to the first data group block (DGBLOCK)
    * comment_addr - Pointer to the measurement file comment text (TXBLOCK) (NIL allowed)
    * program_addr - Pointer to program block (PRBLOCK) (NIL allowed)
    * dg_nr - Number of data groups (redundant information)
    * date - Date at which the recording was started in "DD:MM:YYYY" format
    * time - Time at which the recording was started in "HH:MM:SS" format
    * author - author name
    * organization - organization
    * project - project name
    * subject - subject

    Since version 3.2 the following extra keys were added:

    * abs_time - Time stamp at which recording was started in nanoseconds.
    * tz_offset - UTC time offset in hours (= GMT time zone)
    * time_quality - Time quality class
    * timer_identification - Timer identification (time source),

    Parameters
    ----------
    file_stream : file handle
        mdf file handle

    '''
    def __init__(self, **kargs):
        super().__init__()

        self.address = 64
        try:

            stream = kargs['file_stream']
            stream.seek(64, SEEK_START)

            (self['id'],
             self['block_len'],
             self['first_dg_addr'],
             self['comment_addr'],
             self['program_addr'],
             self['dg_nr'],
             self['date'],
             self['time'],
             self['author'],
             self['organization'],
             self['project'],
             self['subject']) = unpack(HEADER_COMMON_FMT, stream.read(HEADER_COMMON_SIZE))

            if self['block_len'] > HEADER_COMMON_SIZE:
                (self['abs_time'],
                 self['tz_offset'],
                 self['time_quality'],
                 self['timer_identification']) = unpack(HEADER_POST_320_EXTRA_FMT, stream.read(HEADER_POST_320_EXTRA_SIZE))

        except KeyError:
            version = kargs.get('version', 320)
            self['id'] = 'HD'.encode('latin-1')
            self['block_len'] = 208 if version >= 320 else 164
            self['first_dg_addr'] = 0
            self['comment_addr'] = 0
            self['program_addr'] = 0
            self['dg_nr'] = 0
            t1 = time.time() * 10**9
            t2 = time.gmtime()
            self['date'] = '{:\x00<10}'.format(time.strftime('%d:%m:%Y', t2)).encode('latin-1')
            self['time'] = '{:\x00<8}'.format(time.strftime('%X', t2)).encode('latin-1')
            self['author'] = '{:\x00<32}'.format(os.getlogin()).encode('latin-1')
            self['organization'] = '{:\x00<32}'.format('').encode('latin-1')
            self['project'] = '{:\x00<32}'.format('').encode('latin-1')
            self['subject'] = '{:\x00<32}'.format('').encode('latin-1')

            if version >= 320:
                self['abs_time'] = int(t1)
                self['tz_offset'] = 2
                self['time_quality'] = 0
                self['timer_identification'] = '{:\x00<32}'.format('Local PC Reference Time').encode('latin-1')

    def __bytes__(self):
        fmt = HEADER_COMMON_FMT
        keys = HEADER_COMMON_KEYS
        if self['block_len'] > HEADER_COMMON_SIZE:
            fmt += HEADER_POST_320_EXTRA_FMT
            keys += HEADER_POST_320_EXTRA_KEYS
        return pack(fmt, *[self[key] for key in keys])


class ProgramBlock(dict):
    ''' PRBLOCK class derived from *dict*

    The ProgramBlock object can be created in two modes:

    * using the *file_stream* and *address* keyword parameters - when reading from file
    * using any of the following presented keys - when creating a new ProgramBlock

    The keys have the following meaning:

    * id - Block type identifier, always "PR"
    * block_len - Block size of this block in bytes (entire PRBLOCK)
    * data - Program-specific data

    Parameters
    ----------
    file_stream : file handle
        mdf file handle
    address : int
        block address inside mdf file

    Attributes
    ----------
    size : int
        size of bytes reprezentation of PRBLOCK
    address : int
        block address inside mdf file

    '''
    def __init__(self, **kargs):
        super().__init__()

        try:
            stream = kargs['file_stream']
            self.address = address = kargs['address']
            stream.seek(address, SEEK_START)

            (self['id'],
             self['block_len']) = unpack('<2sH', stream.read(4))
            self['data'] = stream.read(self['block_len'] - 4)

        except KeyError:
            pass

    def __bytes__(self):
        fmt = FMT_PROGRAM_BLOCK.format(self['block_len'])
        return pack(fmt, *[self[key] for key in KEYS_PROGRAM_BLOCK])


class SampleReduction(dict):
    ''' SRBLOCK class derived from *dict*

    Currently the SampleReduction object can only be created by using the *file_stream* and *address* keyword parameters - when reading from file

    The keys have the following meaning:

    * id - Block type identifier, always "SR"
    * block_len - Block size of this block in bytes (entire SRBLOCK)
    * next_sr_addr - Pointer to next sample reduction block (SRBLOCK) (NIL allowed)
    * data_block_addr - Pointer to the data block for this sample reduction
    * cycles_nr - Number of reduced samples in the data block.
    * time_interval - Length of time interval [s] used to calculate the reduced samples.

    Parameters
    ----------
    file_stream : file handle
        mdf file handle
    address : int
        block address inside mdf file

    Attributes
    ----------
    size : int
        size of bytes reprezentation of SRBLOCK
    address : int
        block address inside mdf file

    '''
    def __init__(self, **kargs):
        super().__init__()

        try:
            stream = kargs['file_stream']
            self.address = address = kargs['address']
            stream.seek(address, SEEK_START)

            (self['id'],
             self['block_len'],
             self['next_sr_addr'],
             self['data_block_addr'],
             self['cycles_nr'],
             self['time_interval']) = unpack(FMT_SAMPLE_REDUCTION_BLOCK, stream.read(SR_BLOCK_SIZE))

        except KeyError:
            pass

    def __bytes__(self):
        return pack(FMT_SAMPLE_REDUCTION_BLOCK, *[self[key] for key in KEYS_SAMPLE_REDUCTION_BLOCK])


class TextBlock(dict):
    ''' TXBLOCK class derived from *dict*

    The ProgramBlock object can be created in two modes:

    * using the *file_stream* and *address* keyword parameters - when reading from file
    * using the classmethod *from_text*

    The keys have the following meaning:

    * id - Block type identifier, always "TX"
    * block_len - Block size of this block in bytes (entire TXBLOCK)
    * text - Text (new line indicated by CR and LF; end of text indicated by 0)

    Parameters
    ----------
    file_stream : file handle
        mdf file handle
    address : int
        block address inside mdf file
    text : bytes
        bytes for creating a new TextBlock

    Attributes
    ----------
    size : int
        size of bytes reprezentation of TXBLOCK
    address : int
        block address inside mdf file
    text_str : str
        text data as unicode string

    Examples
    --------
    >>> tx1 = TextBlock.from_text('VehicleSpeed')
    >>> tx1.text_str
    'VehicleSpeed'
    >>> tx1['text']
    b'VehicleSpeed'

    '''
    def __init__(self, **kargs):
        super().__init__()
        try:
            stream = kargs['file_stream']
            self.address = address = kargs['address']

            stream.seek(address, SEEK_START)
            (self['id'],
             self['block_len']) = unpack('<2sH', stream.read(4))
            size = self['block_len'] - 4
            self['text'] = text = unpack('<{}s'.format(size), stream.read(size))[0]

            self.text_str = text.decode('latin-1').replace('\0', '')
        except KeyError:
            self.address = 0
            text = kargs.get('text', b'empty')
            text_length = len(text)
            align = text_length % 8
            if align == 0 and text[-1] == b'\x00':
                padding = 0
            else:
                padding = 8 - align

            self['id'] = kargs.get('id', b'TX')
            self['block_len'] = text_length + padding + 4

            self.text_str = text.decode('latin-1').replace('\0', '')
            self['text'] = text + b'\00' * padding

    @classmethod
    def from_text(cls, text):
        """
        Creates a *TextBlock* object from a string or bytes

        Parameters
        ----------
        text : str | bytes
            input string

        """

        if isinstance(text, str):
            text = text.encode('latin-1')
        elif isinstance(text, bytes):
            pass
        else:
            raise TypeError('Expected str or bytes object; got "{}" of type {}'.format(text, type(text)))
        return cls(text=text)

    def __bytes__(self):
        #return pack('<2sH{}s'.format(self.size - 4), *[self[key] for key in KEYS_TEXT_BLOCK])
        # for performance reasons:
        return pack('<2sH' + str(self['block_len']-4) + 's', *[self[key] for key in KEYS_TEXT_BLOCK])


class TriggerBlock(dict):
    ''' TRBLOCK class derived from *dict*

    The TriggerBlock object can be created in two modes:

    * using the *file_stream* and *address* keyword parameters - when reading from file
    * using the classmethod *from_text*

    The keys have the following meaning:

    * id - Block type identifier, always "TX"
    * block_len - Block size of this block in bytes (entire TRBLOCK)
    * text_addr - Pointer to trigger comment text (TXBLOCK) (NIL allowed)
    * trigger_events_nr - Number of trigger events n (0 allowed)
    * trigger_{n}_time - Trigger time [s] of trigger event *n*
    * trigger_{n}_pretime - Pre trigger time [s] of trigger event *n*
    * trigger_{n}_posttime - Post trigger time [s] of trigger event *n*

    Parameters
    ----------
    file_stream : file handle
        mdf file handle
    address : int
        block address inside mdf file

    Attributes
    ----------
    size : int
        size of bytes reprezentation of TRBLOCK
    address : int
        block address inside mdf file

    '''
    def __init__(self, **kargs):
        super().__init__()

        try:
            self.address = address = kargs['address']
            stream = kargs['file_stream']

            stream.seek(address + 2, SEEK_START)
            size = unpack('<H', stream.read(2))[0]
            stream.seek(address, SEEK_START)
            block = stream.read(size)

            (self['id'],
             self['block_len'],
             self['text_addr'],
             self['trigger_events_nr']) = unpack('<2sHIH', block[:10])

            for i, (t, pre, post) in enumerate(iter_unpack('<3d', block[10:])):
                (self['trigger_{}_time'.format(i)],
                 self['trigger_{}_pretime'.format(i)],
                 self['trigger_{}_posttime'.format(i)]) = t, pre, post

        except KeyError:
            pass

    def __bytes__(self):
        triggers_nr = self['trigger_events_nr']
        fmt = '<2sHIH{}d'.format(triggers_nr * 3)
        keys = ('id', 'block_len', 'text_addr', 'trigger_events_nr')
        for i in range(triggers_nr):
            keys += ('trigger_{}_time'.format(i), 'trigger_{}_pretime'.format(i), 'trigger_{}_posttime'.format(i))
        return pack(fmt, *[self[key] for key in keys])


if __name__ == '__main__':
    pass

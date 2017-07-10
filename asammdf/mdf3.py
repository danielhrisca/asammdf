"""
ASAM MDF version 3 file format module

"""

from __future__ import absolute_import, division, print_function
import time

from numpy import interp, linspace, dtype, amin, amax
from numpy import array, searchsorted, log, exp, clip
from numexpr import evaluate
from numpy.core.records import fromstring, fromarrays

from struct import unpack, pack
from collections import OrderedDict, defaultdict
import os
import os.path

import itertools
from functools import lru_cache

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

HEADER_300_SIZE = 228
HEADER_320_EXTRA_SIZE = 44
HEADER_FMT = '<8s8s8s4H2s30s2sH3IH10s8s32s32s32s32s'
HEADER_320_EXTRA_FMT = 'Q2H32s'

TIME_FAC = 10 ** -9
TIME_CH_SIZE = 8
SI_BLOCK_SIZE = 128
FH_BLOCK_SIZE = 56
DG31_BLOCK_SIZE = 24
DG32_BLOCK_SIZE = 28
HD_BLOCK_SIZE = 104
CN_BLOCK_SIZE = 228
CG_BLOCK_SIZE = 26
DT_BLOCK_SIZE = 24
CC_COMMON_BLOCK_SIZE = 46
CC_ALG_BLOCK_SIZE = 88
CC_LIN_BLOCK_SIZE = 62
CC_POLY_BLOCK_SIZE = 94
CC_EXPO_BLOCK_SIZE = 102
CC_FORMULA_BLOCK_SIZE = 304

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


def fmt(data_type, size):
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
        return 'b'
    if data_type in (DATA_TYPE_UNSIGNED_INTEL, DATA_TYPE_UNSIGNED):
        return '<u{}'.format(size)
    elif data_type == DATA_TYPE_UNSIGNED_MOTOROLA:
        return '>u{}'.format(size)
    elif data_type in (DATA_TYPE_SIGNED_INTEL, DATA_TYPE_SIGNED):
        return '<i{}'.format(size)
    elif data_type == DATA_TYPE_SIGNED_MOTOROLA:
        return '>i{}'.format(size)
    elif data_type in (DATA_TYPE_FLOAT, DATA_TYPE_DOUBLE, DATA_TYPE_FLOAT_INTEL, DATA_TYPE_DOUBLE_INTEL):
        return '<f{}'.format(size)
    elif data_type in (DATA_TYPE_FLOAT_MOTOROLA, DATA_TYPE_DOUBLE_MOTOROLA):
        return '>f{}'.format(size)
    elif data_type == DATA_TYPE_STRING:
        return 'a{}'.format(size)


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


class MDF3(object):
    """

    Parameters
    ----------
    file_name : string
        mdf file name
    empty : bool
        set to *True* if creating a new file from scratch; default *False*
    load_measured_data : bool
        load data option; default *True* ::

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
    def __init__(self, file_name, empty=False, load_measured_data=True,  compression=False, version='3.20'):
        self.groups = []
        self.header = OrderedDict()
        self.file_history = None
        self.name = file_name
        self.load_measured_data = load_measured_data
        self.compression = compression
        self.channel_db = {}
        self.masters_db = {}

        if not empty:
            self._read()
            self.is_new = False
        else:
            self.load_measured_data = True
            self.is_new = True

            self.groups = []

            self.header['file_identification'] = 'MDF     '.encode('latin-1')
            self.header['version_str'] = version.encode('latin-1')
            self.header['program_identification'] = 'Python  '.encode('latin-1')
            # default to Intel byteorder
            self.header['byte_order'] = 0

            self.byteorder = '<'

            self.header['float_format'] = 0
            self.header['mdf_version'] = int(float(version) * 100)
            self.version = self.header['mdf_version']
            self.header['reserved0'] = 0
            self.header['reserved1'] = b'\x00' * 2
            self.header['reserved2'] = b'\x00' * 30
            # Header block
            self.header['id'] = 'HD'.encode('latin-1')
            self.header['block_len'] = 208 if version == '3.20' else 164
            self.header['first_dg_addr'] = 0
            self.header['comment_addr'] = 0
            self.header['program_addr'] = 0
            self.header['dg_nr'] = 0
            t1 = time.time() * 10**9
            t2 = time.gmtime()
            self.header['date'] = '{:\x00<10}'.format(time.strftime('%d:%m:%Y', t2))
            self.header['time'] = '{:\x00<8}'.format(time.strftime('%X', t2))
            self.header['author'] = '{:\x00<32}'.format(os.getlogin())
            self.header['organization'] = '{:\x00<32}'.format('ST')
            self.header['project'] = '{:\x00<32}'.format('')
            self.header['subject'] = '{:\x00<32}'.format('')
            if self.version == 320:
                self.header['abs_time'] = int(t1)
                self.header['tz_offset'] = 2
                self.header['time_quality'] = 0
                self.header['timer_identification'] = '{:\x00<32}'.format('Local PC Reference Time')
            self.file_history = TextBlock.from_text('''<FHcomment>
<TX>created</TX>
<tool_id>PythonMDFEditor</tool_id>
<tool_vendor> </tool_vendor>
<tool_version>1.0</tool_version>
</FHcomment>''')

    def append(self, signals, t, signal_names, signal_units, info):
        """
        Appends a new data group.

        Parameters
        ----------
        signals : list
            list on numpy.array signals
        t : numpy.array
            time channel
        signal_names : list
            list of signal name strings
        signal_units : list
            list of signal unit strings
        info : dict
            acquisition information; the 'conv' key contains a dict of conversions, each key being
            the signal name for which the conversion is specified


        """

        self.groups.append({})
        dg_cntr = len(self.groups) - 1
        gp = self.groups[-1]

        channel_nr = len(signals)
        cycles_nr = len(t)

        t_type, t_size = fmt_to_datatype(t.dtype)

        gp['defined_channels'] = signal_names
        gp['channels'] = gp_channels = []
        gp['channel_conversions'] = gp_conv = []
        gp['channel_sources'] = gp_source = []
        gp['texts'] = gp_texts = {'channels': [],
                       'conversion_vtabr': [],
                       'channel_group': []}

        #time channel texts
        for _, item in gp_texts.items():
            item.append({})

        gp_texts['channel_group'][-1]['comment_addr'] = TextBlock.from_text(
            info.get('acq_comment_addr', 'Python').encode('latin-1'))

        #channels texts
        for name in signal_names:
            for _, item in gp['texts'].items():
                item.append({})
            if len(name) >= 32:
                gp_texts['channels'][-1]['long_name_addr'] = TextBlock.from_text(
                    name.encode('latin-1'))

        #conversion for time channel
        kargs = {'conversion_type': CONVERSION_TYPE_NONE,
                 'unit': 's'.encode('latin-1'),
                 'min_phy_value': t[0],
                 'max_phy_value': t[-1]}
        gp_conv.append(ChannelConversion(**kargs))

        min_max = [(amin(signal), amax(signal)) for signal in signals]
        #conversion for channels
        for idx, (sig, unit) in enumerate(zip(signals, signal_units)):
            #print(sig.dtype.char,signal_names[idx],info['conv'][signal_names[idx]])
            try:
                descriptors = info['conv'][signal_names[idx]]
                dim = len(descriptors)
                if dim == 2:
                    kargs = {}
                    kargs['conversion_type'] = CONVERSION_TYPE_VTAB
                    raw = descriptors[0]
                    phys = descriptors[1]
                    for i, (r_, p_) in enumerate(zip(raw, phys)):
                        kargs['text_{}'.format(i)] = p_[:31] + b'\x00'
                        kargs['param_val_{}'.format(i)] = r_
                    kargs['ref_param_nr'] = len(raw)
                    kargs['unit'] = unit.encode('latin-1')
                    signals[idx] = sig
                elif dim == 3:
                    kargs = {}
                    kargs['conversion_type'] = CONVERSION_TYPE_VTABR
                    lower = descriptors[0]
                    upper = descriptors[1]
                    texts = descriptors[2]
                    kargs['unit'] = unit.encode('latin-1')
                    kargs['ref_param_nr'] = len(upper)

                    for i, (u_, l_, t_) in enumerate(zip(upper, lower, texts)):
                        kargs['lower_{}'.format(i)] = l_
                        kargs['upper_{}'.format(i)] = u_
                        kargs['text_{}'.format(i)] = 0
                        gp_texts['conversion_vtabr'][-1]['text_{}'.format(i)] = TextBlock.from_text(t_)

                    signals[idx] = sig
                else:
                    raise ValueError('{} arrays were given for conversion; must be 2 for VTAB or 3 for VTABR'.format(dim))
                gp_conv.append(ChannelConversion(**kargs))
            except KeyError:
                kargs = {'conversion_type': CONVERSION_TYPE_NONE,
                 'unit': signal_units[idx].encode('latin-1') if signal_units[idx] else b'',
                 'min_phy_value': min_max[idx][0],
                 'max_phy_value': min_max[idx][1]}
                gp_conv.append(ChannelConversion(**kargs))
            except TypeError:
                gp_conv.append(None)

        #source for channels and time
        for i in range(channel_nr + 1):
            kargs = {'module_nr': 0,
                     'module_address': 0,
                     'type': SOURCE_ECU,
                     'description': 'Channel inserted by Python Script'.encode('latin-1')}
            gp_source.append(SourceInformation(**kargs))

        #time channel
        kargs = {'short_name': 't'.encode('latin-1'),
                 #'source_depend_addr': gp['texts']['sources'][0]['path_addr'].address,
                 'channel_type': CHANNEL_TYPE_MASTER,
                 'data_type': t_type,
                 'start_offset': 0,
                 'bit_count': t_size}
        gp_channels.append(Channel(**kargs))

        sig_dtypes = [sig.dtype for sig in signals]
        sig_formats = [fmt_to_datatype(typ) for typ in sig_dtypes]


        #channels
        offset = t_size
        ch_cntr = 1
        for (sigmin, sigmax), (sig_type, sig_size), name in zip(min_max, sig_formats, signal_names):
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

    def display_channel_list(self):
        """
        Displays a complete channel list table to stdout

        """
        print('{:<60}|{:>10}|{:<20}'.format('Channel name',
                                            'Samples',
                                            'Source'))

        for gp in self.groups:
            samples = gp['channel_group']['cycles_nr']
            print('+'.join(('-'*60, '-'*10, '-'*20)))
            for source, channel_name in zip(gp['channel_sources'][1:], gp['defined_channels'][1:]):
                if source:
                    if source['type'] == SOURCE_ECU:
                        source_info = source['ECU_identification'].decode('latin-1').strip()
                    else:
                        source_info = 'CAN: {} {}'.format(hex(source['CAN_id']), source['message_name'].decode('latin-1').strip())
                else:
                    source_info = r'N/A'
                print('{:<60}|{:>10}|{:<20}'.format(channel_name, samples, source_info))

    def _update_addresses(self):
        """
        Update internal block addresses based on added/removed items

        Returns
        -------
        blocks : list
            list of blocks

        """
        #store unique texts and their addresses
        defined_texts = {}

        blocks = []

        additional_bytes = []

        address = HEADER_300_SIZE
        if self.version == 320:
            address += HEADER_320_EXTRA_SIZE
            additional_bytes.append(0)
        else:
            # Pre 3.20 MDF header needs 4 extra bytes for 8byte alignment
            address += 4
            additional_bytes.append(4)

        self.file_history.address = address
        address += self.file_history.size
        # TextBlocks are 8byte aligned
        additional_bytes.append(0)
        blocks.append(self.file_history)

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
                            address += my_dict[key].size
                            # TextBlocks are 8byte aligned
                            additional_bytes.append(0)
                            blocks.append(my_dict[key])

            # ChannelConversions
            cc = gp['channel_conversions']
            for i, conv in enumerate(cc):
                if conv:
                    conv.address = address
                    address += conv.size
                    align = address % 8
                    if align:
                        add = 8 - align
                        address += add
                    else:
                        add = 0
                    additional_bytes.append(add)
                    blocks.append(conv)

                    if conv['conversion_type'] == CONVERSION_TYPE_VTABR:
                        for key, item in gp_texts['conversion_vtabr'][i].items():
                            conv[key] = item.address

            # ChannelSources
            cs = gp['channel_sources']
            for source in cs:
                if source:
                    source.address = address
                    address += source.size
                    # SourceInformation are 8byte aligned
                    additional_bytes.append(0)
                    blocks.append(source)

            # Channels
            ch_texts = gp_texts['channels']
            for i, channel in enumerate(gp['channels']):
                channel.address = address
                # Channels need 4 extra bytes for 8byte alignment
                channel_align_bytes = 4
                address += channel.size + channel_align_bytes
                additional_bytes.append(channel_align_bytes)
                blocks.append(channel)

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
            next_channel['next_ch_addr'] = 0

            # ChannelGroup
            cg = gp['channel_group']
            cg.address = address
            # ChannelGroup needs 6 extra bytes for 8byte alignment
            address += cg.size + 6
            additional_bytes.append(6)
            blocks.append(cg)
            cg['first_ch_addr'] = gp['channels'][0].address
            cg['next_cg_addr'] = 0
            if 'comment_addr' in gp['texts']['channel_group'][0]:
                cg['comment_addr'] = gp_texts['channel_group'][0]['comment_addr'].address

            # DataBlock
            db = gp['data_block']
            db.address = address
            address += db.size
            align = address % 8
            if align:
                add = 8 - align
                address += add
            else:
                add = 0
            additional_bytes.append(add)
            blocks.append(db)

            # DataGroup
            dg = gp['data_group']
            dg.address = address
            address += dg.size
            align = address % 8
            if align:
                add = 8 - align
                address += add
            else:
                add = 0
            additional_bytes.append(add)
            blocks.append(dg)
            dg['first_cg_addr'] = cg.address
            dg['data_block_addr'] = db.address

        for dg, next_dg in pair(self.groups):
            dg['data_group']['next_dg_addr'] = next_dg['data_group'].address

        if self.groups:
            self.header['first_dg_addr'] = self.groups[0]['data_group'].address
            self.header['dg_nr'] = len(self.groups)
            self.header['comment_addr'] = self.file_history.address
            self.header['program_addr'] = 0

        return blocks, additional_bytes

    def get_master_channel(self, gp_nr):
        gp = self.groups[gp_nr]

        for ch, conv in zip(gp['channels'], gp['channel_conversions']):
            if ch['channel_type'] == CHANNEL_TYPE_MASTER:
                time_ch, time_conv = ch, conv
                break

        time_size = time_ch['bit_count'] // 8
        t_fmt = fmt(time_ch['data_type'], time_size)
        block_size = gp['channel_group']['samples_byte_nr'] - gp['data_group']['record_id_nr']

        data = gp['data_block']['data']

        types = dtype([('t', t_fmt),
                   ('res1', 'a{}'.format(block_size - time_size))])

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

        return t

    def get_signal_by_name(self, channel_name, raster=None):
        """
        Gets a channel by name. If the raster is not *None* the output is interpolated accordingly

        Parameters
        ----------
        channel_name : string
            name of channel
        raster : float
            time raster

        Returns
        -------
        vals, t, unit, conversion : (numpy.array, numpy.array, string, tuple)
            The conversion is *None* exept for the VTAB and VTABR conversions

        Raises
        ------
        NameError : if the channel name is not found or if the name if 't'

        """
        if channel_name == 't':
            raise NameError("Can't return the time channel")

        try:
            gp_nr, ch_nr = self.channel_db[channel_name]
        except KeyError:
            raise NameError('Channel "{}" not found'.format(channel_name))

        gp = self.groups[gp_nr]
        channel = gp['channels'][ch_nr]
        conversion = gp['channel_conversions'][ch_nr]

        time_idx = self.masters_db[gp_nr]
        time_ch = gp['channels'][time_idx]
        time_conv = gp['channel_conversions'][time_idx]

        group = gp

        time_size = time_ch['bit_count'] // 8
        t_fmt = fmt(time_ch['data_type'], time_size)
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
        ch_fmt = fmt(channel['data_type'], size)

        if not self.load_measured_data:
            with open(self.name, 'rb') as file_stream:
                # go to the first data block of the current data group
                dat_addr = group['data_group']['data_block_addr']
                read_size = group['channel_group']['samples_byte_nr'] * group['channel_group']['cycles_nr']
                data = DataBlock(file_stream=file_stream, address=dat_addr, size=read_size)['data']
        else:
            data = group['data_block']['data']

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
            ''' if texts are needed
            idx = searchsorted(raw, vals)
            vals = phys[idx]
            '''

        elif conversion_type == CONVERSION_TYPE_VTABR:
            nr = conversion['ref_param_nr']

            texts = array([gp['texts']['conversion_vtabr'][idx]['text_{}'.format(i)]['text'] for i in range(nr)])
            lower = array([conversion['lower_{}'.format(i)] for i in range(nr)])
            upper = array([conversion['upper_{}'.format(i)] for i in range(nr)])
            vals = values['vals']
            ''' if texts are needed
            if 'f' in ch_fmt:
                idx = fromiter((searchsorted(lower, e) for e in vals), uint16)
            else:
                idx = fromiter((searchsorted(lower, e) for e in vals), uint16)
            vals = texts[idx]
            '''

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

        if conversion:
            unit = conversion['unit'].decode('latin-1').strip('\x00')
        else:
            unit = ''

        if conversion_type == CONVERSION_TYPE_VTAB:
            return vals, t, unit, (raw, phys)
        elif conversion_type == CONVERSION_TYPE_VTABR:
            return vals, t, unit, (lower, upper, texts)
        else:
            return vals, t, unit, None

    def _read(self):
        with open(self.name, 'rb') as file_stream:

            # performance optimization
            read = file_stream.read
            seek = file_stream.seek

            dg_cntr = 0
            seek(0, SEEK_START)

            # ID and Header blocks
            (self.header['file_identification'],
            self.header['version_str'],
            self.header['program_identification'],
            self.header['byte_order'],
            self.header['float_format'],
            self.header['mdf_version'],
            self.header['reserved0'],
            self.header['reserved1'],
            self.header['reserved2'],
            self.header['id'],
            self.header['block_len'],
            self.header['first_dg_addr'],
            self.header['comment_addr'],
            self.header['program_addr'],
            self.header['dg_nr'],
            self.header['date'],
            self.header['time'],
            self.header['author'],
            self.header['organization'],
            self.header['project'],
            self.header['subject']) = (unpack(HEADER_FMT, read(HEADER_300_SIZE)))

            self.byteorder = '<' if self.header['byte_order'] == 0 else '>'

            self.version = self.header['mdf_version']
            if self.version == 320:
                (self.header['abs_time'],
                self.header['tz_offset'],
                self.header['time_quality'],
                self.header['timer_identification']) = unpack(HEADER_320_EXTRA_FMT, read(HEADER_320_EXTRA_SIZE))

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
                    grp['defined_channels'] = []
                    grp['channels'] = []
                    grp['channel_conversions'] = []
                    grp['channel_sources'] = []
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
                    defined_channels = grp['defined_channels']
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
                            for i in range(new_conv['ref_param_nr']):
                                address = new_conv['text_{}'.format(i)]
                                if address:
                                    vtab_texts['text_{}'.format(i)] = TextBlock(address=address, file_stream=file_stream)
                        grp['texts']['conversion_vtabr'].append(vtab_texts)


                        # read source block and create source infromation object
                        address = new_ch['source_depend_addr']
                        if address:
                            grp['channel_sources'].append(SourceInformation(address=address, file_stream=file_stream))
                        else:
                            grp['channel_sources'].append(None)

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
                        #if 'display_name_addr' in grp['texts']['channels'][-1]:
                       #     grp['channels'][-1].display_name = grp['texts']['channels'][-1]['display_name_addr']['text'].decode('latin-1').strip('\x00')
                        defined_channels.append(new_ch.name)

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

    def remove_channel(self, channel_name):
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
                if channel_name in gp['defined_channels']:
                    # if this is the only channel in the channel group
                    print('remove', channel_name, i)
                    if len(gp['defined_channels']) == 2:
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
                        j = gp['defined_channels'].index(channel_name)
                        gp['defined_channels'].pop(j)
                        # remove all text blocks associated with the channel
                        for key in ('channels', 'conversion_vtabr'):
                            gp['texts'][key].pop(j)

                        #print(hex(id(gp)), hex(id(self.groups[i])))

                        channel = gp['channels'].pop(j)
                        start_offset = channel['start_offset']
                        bit_count = channel['bit_count']
                        byte_offset, bit_offset = divmod(channel['start_offset'], 8)
                        byte_size = channel['bit_count'] // 8

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
                        gp['channel_sources'].pop(j)

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
                gp_nr, ch_nr = self.channel_db[channel_name]
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

    def save_to_file(self, new_file_name=None):
        """
        Saves the measurement to disk

        Parameters
        ----------
        new_file_name : string
            the file name used for saving the measurement; if not supplied the original measurement is overwritten

        """
        if not self.load_measured_data:
            print('Not implemented')
            return

        if new_file_name:
            out_file_name = new_file_name.replace('.mdf', '')
            out_file_name = os.path.join(os.path.dirname(self.name), out_file_name + '.mdf')
        else:
            out_file_name = self.name

        with open(out_file_name, 'wb') as new_file:
            blocks, additional_bytes = self._update_addresses()
            # update header information
            header_fmt = HEADER_FMT
            if self.version == 320:
                header_fmt += HEADER_320_EXTRA_FMT

            header_items = [val for k, val in self.header.items()]
            for i, val in enumerate(header_items):
                if isinstance(val, str):
                    header_items[i] = val.encode('latin-1')

            new_file.write(pack(header_fmt, *header_items))
            add_bytes = additional_bytes[0]
            if add_bytes:
                new_file.write(b'\x00' * add_bytes)

            for block, add_bytes in zip(blocks, additional_bytes[1:]):
                new_file.write(bytes(block))
                if add_bytes:
                    new_file.write(b'\x00' * add_bytes)


    def time_shift_group(self, channel_name, offset):
        """
        Shift a data group's time channel.

        Parameters
        ----------
        channel_name : string
            a channel name inside the data group used to identify the data group
        offset : float
            time shift offset

        """
        for group in self.groups:
            if channel_name in group['defined_channels']:
                block_size = group['channel_group']['samples_byte_nr'] -\
                             group['data_group']['record_id_nr']
                time_ch = group['channels'][0]
                time_conv = group['channel_conversions'][0]
                time_size = time_ch['bit_count'] // 8
                time_data_type = time_ch['data_type']
                time_conv_type = CONVERSION_TYPE_NONE if time_conv is None else time_conv['conversion_type']
                if time_conv_type == CONVERSION_TYPE_LINEAR:
                    time_a = time_conv['a']
                    time_b = time_conv['b']
                t_fmt = fmt(time_data_type, time_size)

                types = dtype([('t', t_fmt),
                   ('res1', 'a{}'.format(block_size - time_size))])

                data = group['data_block']['data']
                values = fromstring(data, types)

                if time_conv_type == CONVERSION_TYPE_LINEAR:
                    t = values['t'] * time_a + time_b
                    t = (t + offset - time_b ) / time_a
                else:
                    t = values['t']
                    t = t + offset
                vals = values['res1']

                group['data_block']['data'] = fromarrays((t, vals), dtype=types).tostring()
                break
        else:
            print('{} not found in {}'.format(channel_name, self.name))


class Channel(dict):

    def __init__(self, *args, **kargs):
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

        self.size = self['block_len']

    def __bytes__(self):
        return pack(FMT_CHANNEL, *[self[key] for key in KEYS_CHANNEL])


class ChannelGroup(dict):
    def __init__(self, *args, **kargs):
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
        self.size = self['block_len']

    def __bytes__(self):
        return pack(FMT_CHANNEL_GROUP, *[self[key] for key in KEYS_CHANNEL_GROUP])


class ChannelConversion(dict):
    def __init__(self, *args, **kargs):
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
                print(hex(address))
                self['formula'] = unpack('<{}s'.format(size - 46), block[CC_COMMON_BLOCK_SIZE:])[0]

            elif conv_type in (CONVERSION_TYPE_TABI, CONVERSION_TYPE_TABX):
                nr = self['ref_param_nr']

                for i in range(nr):
                    (self['raw_{}'.format(i)],
                    self['phys_{}'.format(i)]) = unpack('<2d', block[CC_COMMON_BLOCK_SIZE + i*16: CC_COMMON_BLOCK_SIZE + (i+1)*16])

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

                for i in range(nr):
                    (self['param_val_{}'.format(i)],
                    self['text_{}'.format(i)]) = unpack('<d32s', block[CC_COMMON_BLOCK_SIZE + i*40: CC_COMMON_BLOCK_SIZE + (i + 1)*40])

            elif conv_type == CONVERSION_TYPE_VTABR:
                nr = self['ref_param_nr']

                for i in range(nr):
                    (self['lower_{}'.format(i)],
                    self['upper_{}'.format(i)],
                    self['text_{}'.format(i)]) = unpack('<2dI', block[CC_COMMON_BLOCK_SIZE + i*20: CC_COMMON_BLOCK_SIZE + (i + 1)*20])
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
        self.size = self['block_len']

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
            if not self.size == CC_LIN_BLOCK_SIZE:
                fmt += '{}s'.format(self.size - CC_LIN_BLOCK_SIZE)
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


class DataBlock(dict):
    def __init__(self, *args, **kargs):
        """
        Creates a *Data Block* object from a block loaded from a measurement file

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
        super().__init__()

        self.size = 0
        try:
            stream = kargs['file_stream']
            size = kargs['size']
            self.address = address = kargs['address']
            stream.seek(address, SEEK_START)

            self.compression = kargs.get('compression', False)
            self['data'] = stream.read(size)


        except KeyError as err:
            self.address = 0
            self.compression = kargs.get('compression', False)
            self['data'] = kargs.get('data', bytes())

    def __setitem__(self, item, value):
        if item == 'data':
            self.size = len(value)
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
    def __init__(self, *args, **kargs):
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
            self.size = self['block_len']
            self['next_dg_addr'] = kargs.get('next_dg_addr', 0)
            self['first_cg_addr'] = kargs.get('first_cg_addr', 0)
            self['trigger_addr'] = kargs.get('comment_addr', 0)
            self['data_block_addr'] = kargs.get('data_block_addr', 0)
            self['cg_nr'] = kargs.get('cg_nr', 1)
            self['record_id_nr'] = kargs.get('record_id_nr', 0)
            if self.size == DG32_BLOCK_SIZE:
                self['reserved0'] = b'\x00\x00\x00\x00'
        self.size = self['block_len']

    def __bytes__(self):
        if self.size == DG32_BLOCK_SIZE:
            fmt = FMT_DATA_GROUP_32
            keys = KEYS_DATA_GROUP_32
        else:
            fmt = FMT_DATA_GROUP
            keys = KEYS_DATA_GROUP
        return pack(fmt, *[self[key] for key in keys])


class ProgramBlock(OrderedDict):
    def __init__(self, *args, **kargs):
        super().__init__()

        self.address = 0
        self['id'] = kargs.get('id', 'PR'.encode('latin-1'))
        self['block_len'] = kargs.get('block_len', 8)
        self['data'] = kargs.get('data',b'\x00\x00\x00\x00')
        self.size = self['block_len']

    @classmethod
    def from_address(cls, address, stream):
        """
        Creates a *Program specific* object from a block loaded from a measurement file

        Parameters
        ----------
        address : int
            block address inside the measurement file
        stream : file.io.handle
            binary file stream

        """
        stream.seek(address + 2, SEEK_START)
        size = unpack('<H', stream.read(2))[0]
        stream.seek(address, SEEK_START)
        block = stream.read(size)

        kargs = {}

        (kargs['id'],
         kargs['block_len'],
         kargs['data']) = unpack('<2sH{}s'.format(size - 4), block)

        return cls(**kargs)

    def __bytes__(self):
        fmt = '<2sH{}s'.format(self.size)
        return pack(fmt, *self.values())


class SourceInformation(dict):
    def __init__(self, *args, **kargs):
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
            self['block_len'] = kargs.get('block_len', SI_BLOCK_SIZE)
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
        self.size = self['block_len']

    def __bytes__(self):
        typ = self['type']
        if typ == SOURCE_ECU:
            fmt = FMT_SOURCE_ECU
            keys = KEYS_SOURCE_ECU
        else:
            fmt = FMT_SOURCE_VECTOR
            keys = KEYS_SOURCE_VECTOR
        return pack(fmt, *[self[key] for key in keys])


class TextBlock(dict):
    def __init__(self, *args, **kargs):
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
        self.size = self['block_len']

    @classmethod
    def from_text(cls, text):
        """
        Creates a *Text Block* object from a string

        Parameters
        ----------
        text : str
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
        return pack('<2sH' + str(self.size-4) + 's', *[self[key] for key in KEYS_TEXT_BLOCK])

class TriggerBlock(OrderedDict):
    def __init__(self, *args, **kargs):
        super().__init__()

        self.address = 0
        self['id'] = kargs.get('id', 'TR'.encode('latin-1'))
        self['block_len'] = kargs.get('block_len', 10)
        self['text_addr'] = kargs.get('text_addr', 0)
        self['trigger_events_nr'] = kargs.get('trigger_events_nr', 0)

        for i in range(self['trigger_events_nr']):
            self['trigger_{}_time'.format(i)] = kargs['trigger_{}_time'.format(i)]
            self['trigger_{}_pretime'.format(i)] = kargs['trigger_{}_pretime'.format(i)]
            self['trigger_{}_posttime'.format(i)] = kargs['trigger_{}_posttime'.format(i)]
        self.size = self['block_len']

    @classmethod
    def from_address(cls, address, stream):
        """
        Creates a *Program specific* object from a block loaded from a measurement file

        Parameters
        ----------
        address : int
            block address inside the measurement file
        stream : file.io.handle
            binary file stream

        """
        stream.seek(address + 2, SEEK_START)
        size = unpack('<H', stream.read(2))[0]
        stream.seek(address, SEEK_START)
        block = stream.read(size)

        kargs = {}

        (kargs['id'],
         kargs['block_len'],
         kargs['text_addr'],
         kargs['trigger_events_nr']) = unpack('<2sHIH', block[:10])

        trigger_nr = kargs['trigger_events_nr']

        for i in range(trigger_nr):
            (kargs['trigger_{}_time'.format(i)],
            kargs['trigger_{}_pretime'.format(i)],
            kargs['trigger_{}_posttime'.format(i)]) = unpack('<3d', block[10 + i*24: 10 + (i + 1)* 24])

        return cls(**kargs)

    def __bytes__(self):
        nr = self['trigger_events_nr']
        fmt = '<2sHIH{}d'.format(nr*3)
        return pack(fmt, *self.values())


if __name__ == '__main__':
    pass

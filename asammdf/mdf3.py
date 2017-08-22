"""
ASAM MDF version 3 file format module

"""
from __future__ import print_function, division
import sys
PYVERSION = sys.version_info[0]

import os
import time
import warnings

from collections import defaultdict
from functools import reduce

from numpy import (interp, linspace, dtype, amin, amax, array_equal,
                   array, searchsorted, log, exp, clip, union1d, float64,
                   uint8, frombuffer, issubdtype, flexible, arange, recarray)
from numpy.core.records import fromstring, fromarrays
from numexpr import evaluate

from .utils import MdfException, get_fmt, pair, fmt_to_datatype
from .signal import Signal
from .v3constants import *
from .v3blocks import (Channel, ChannelConversion, ChannelDependency,
                       ChannelExtension, ChannelGroup, DataBlock, DataGroup,
                       FileIdentificationBlock, HeaderBlock, ProgramBlock,
                       SampleReduction, TextBlock, TriggerBlock)

if PYVERSION == 2:
    def bytes(obj):
        return obj.__bytes__()


__all__ = ['MDF3', ]


class MDF3(object):
    """If the *name* exist it will be loaded otherwise an empty file will be created that can be later saved to disk

    Parameters
    ----------
    name : string
        mdf file name
    load_measured_data : bool
        load data option; default *True*

        * if *True* the data group binary data block will be loaded in RAM
        * if *False* the channel data is read from disk on request

    version : string
        mdf file version ('3.00', '3.10', '3.20' or '3.30'); default '3.20'

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
    version : int
        mdf version
    channels_db : dict
        used for fast channel access by name; for each name key the value is a (group index, channel index) tuple
    masters_db : dict
        used for fast master channel access; for each group index key the value is the master channel index

    """

    def __init__(self, name=None, load_measured_data=True, version='3.20'):
        self.groups = []
        self.header = None
        self.identification = None
        self.file_history = None
        self.name = name
        self.load_measured_data = load_measured_data
        self.channels_db = {}
        self.masters_db = {}

        if name and os.path.isfile(name):
            self._read()
        else:
            self.groups = []

            self.identification = FileIdentificationBlock(version=version)
            self.version = version
            self.header = HeaderBlock(version=self.version)

            self.byteorder = '<'

    def _load_group_data(self, group):
        """ get group's data block bytes when load_measured_data=False """
        with open(self.name, 'rb') as file_stream:
            # go to the first data block of the current data group
            dat_addr = group['data_group']['data_block_addr']

            if group.get('sorted', True):
                read_size = group['channel_group']['samples_byte_nr'] * group['channel_group']['cycles_nr']
                data = DataBlock(file_stream=file_stream, address=dat_addr, size=read_size)['data']

            else:
                read_size = group['size']
                record_id = group['channel_group']['record_id']
                cg_size = group['record_size']
                record_id_nr = group['data_group']['record_id_nr'] if group['data_group']['record_id_nr'] <= 2 else 0
                cg_data = []

                data = DataBlock(file_stream=file_stream, address=dat_addr, size=read_size)['data']

                i = 0
                size = len(data)
                while i < size:
                    rec_id = data[i]
                    # skip redord id
                    i += 1
                    rec_size = cg_size[rec_id]
                    if rec_id == record_id:
                        rec_data = data[i: i+rec_size]
                        cg_data.append(rec_data)
                    # if 2 record id's are sued skip also the second one
                    if record_id_nr == 2:
                        i += 1
                    # go to next record
                    i += rec_size
                data = b''.join(cg_data)
        return data

    def _prepare_record(self, group):
        """ compute record dtype and parents dict fro this group

        Parameters
        ==========
        group : dict
            MDF group dict

        Returns
        =======
        parents, dtypes : dict, numpy.dtype
            mapping of channels to records fields, records fiels dtype

        """
        grp = group
        record_size = grp['channel_group']['samples_byte_nr'] << 3
        next_byte_aligned_position = 0
        types = []
        current_parent = ""
        parent_start_offset = 0
        parents = {}

        for new_ch in sorted(grp['channels']):
            if new_ch['ch_depend_addr']:
                continue

            start_offset = new_ch['start_offset']
            bit_offset = start_offset % 8
            data_type = new_ch['data_type']
            bit_count = new_ch['bit_count']
            name = new_ch.name

            if start_offset >= next_byte_aligned_position:
                parent_start_offset = (start_offset >> 3 ) << 3
                parents[name] = name, bit_offset

                # check if there are byte gaps in the record
                gap = (parent_start_offset - next_byte_aligned_position) >> 3
                if gap:
                    types.append( ('', 'a{}'.format(gap)) )

                # adjust size to 1, 2, 4 or 8 bytes
                size = bit_offset + bit_count
                if data_type != DATA_TYPE_STRING:
                    if size > 32:
                        next_byte_aligned_position = parent_start_offset + 64
                        size = 8
                    elif size > 16:
                        next_byte_aligned_position = parent_start_offset + 32
                        size = 4
                    elif size > 8:
                        next_byte_aligned_position = parent_start_offset + 16
                        size = 2
                    else:
                        next_byte_aligned_position = parent_start_offset + 8
                        size = 1

                types.append( (name, get_fmt(data_type, size)) )

                current_parent = name
            else:
                parents[name] = current_parent, start_offset - parent_start_offset
        gap = (record_size - next_byte_aligned_position) >> 3
        if gap:
            types.append( ('', 'a{}'.format(gap)) )

        return parents, dtype(types)

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

            self.version = self.identification['version_str'].decode('latin-1').strip(' \n\t\x00')

            self.file_history = TextBlock(address=self.header['comment_addr'], file_stream=file_stream)

            # this will hold all channels with channel dependecies
            ch_dep = []
            # this will hold mapping from channel address to Channel object
            ch_map = {}

            parents_map = {}

            # go to first date group
            dg_addr = self.header['first_dg_addr']
            # read each data group sequentially
            while dg_addr:
                gp = DataGroup(address=dg_addr, file_stream=file_stream)
                cg_nr = gp['cg_nr']
                cg_addr = gp['first_cg_addr']
                data_addr = gp['data_block_addr']

                # read trigger information if available
                trigger_addr = gp['trigger_addr']
                if trigger_addr:
                    trigger = TriggerBlock(address=trigger_addr, file_stream=file_stream)
                    if trigger['comment_addr']:
                        trigger_text = TextBlock(address=trigger['comment_addr'], file_stream=file_stream)
                    else:
                        trigger_text = None
                else:
                    trigger = None
                    trigger_text = None

                new_groups = []
                for i in range(cg_nr):

                    new_groups.append({})
                    grp = new_groups[-1]
                    grp['channels'] = []
                    grp['channel_conversions'] = []
                    grp['channel_extensions'] = []
                    grp['data_block'] = None
                    grp['texts'] = {'channels': [], 'conversion_tab': [], 'channel_group': []}
                    grp['trigger'] = [trigger, trigger_text]

                    kargs = {'first_cg_addr': cg_addr,
                             'data_block_addr': data_addr}
                    if self.version in ('3.20', '3.30'):
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

                        # check if it has channeld ependencies
                        if new_ch['ch_depend_addr']:
                            ch_dep.append(new_ch)

                        # update channel map
                        ch_map[ch_addr] = new_ch

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
                        grp['texts']['conversion_tab'].append(vtab_texts)


                        if self.load_measured_data:
                            # read source block and create source infromation object
                            address = new_ch['source_depend_addr']
                            if address:
                                grp['channel_extensions'].append(ChannelExtension(address=address, file_stream=file_stream))
                            else:
                                grp['channel_extensions'].append(None)
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
                            new_ch.name = new_ch['short_name'].decode('latin-1').strip(' \n\t\x00')

                        if new_ch.name in self.channels_db:
                            self.channels_db[new_ch.name].append((dg_cntr, ch_cntr))
                        else:
                            self.channels_db[new_ch.name] = []
                            self.channels_db[new_ch.name].append((dg_cntr, ch_cntr))

                        if new_ch['channel_type'] == CHANNEL_TYPE_MASTER:
                            self.masters_db[dg_cntr] = ch_cntr
                        # go to next channel of the current channel group
                        ch_addr = new_ch['next_ch_addr']
                        ch_cntr += 1
                        grp_chs.append(new_ch)

                    cg_addr = grp['channel_group']['next_cg_addr']
                    dg_cntr += 1

                # store channel groups record sizes dict and data block size in each
                # new group data belong to the initial unsorted group, and add
                # the key 'sorted' with the value False to use a flag;
                # this is used later if load_measured_data=False

                if cg_nr > 1:
                    cg_size = {}
                    size = 0
                    record_id_nr = gp['record_id_nr'] if gp['record_id_nr'] <= 2 else 0
                    for grp in new_groups:
                        size += (grp['channel_group']['samples_byte_nr'] + record_id_nr) * grp['channel_group']['cycles_nr']
                        cg_size[grp['channel_group']['record_id']] = grp['channel_group']['samples_byte_nr']

                    for grp in new_groups:
                        grp['sorted'] = False
                        grp['record_size'] = cg_size
                        grp['size'] = size
                else:
                    record_id_nr = gp['record_id_nr'] if gp['record_id_nr'] <= 2 else 0
                    size = (grp['channel_group']['samples_byte_nr'] + record_id_nr) * grp['channel_group']['cycles_nr']

                if self.load_measured_data:

                    # read data block of the current data group
                    dat_addr = gp['data_block_addr']
                    if dat_addr:
                        seek(dat_addr, SEEK_START)
                        data = read(size)
                    else:
                        data = b''
                    if cg_nr == 1:
                        grp = new_groups[0]
                        kargs = {'data': data}
                        grp['data_block'] = DataBlock(**kargs)

                    else:
                        cg_data = defaultdict(list)
                        i = 0
                        size = len(data)
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
                            grp['channel_group']['record_id'] = 1
                            grp['data_block'] = DataBlock(**kargs)

                self.groups.extend(new_groups)

                # go to next data group
                dg_addr = gp['next_dg_addr']

            for ch in ch_dep:
                dep_blk = ChannelDependency(address=ch['ch_depend_addr'], file_stream=file_stream)
                for i in range(dep_blk['sd_nr']):
                    ch.dependencies.append(ch_map[dep_blk['ch_{}'.format(i)]])

    def add_trigger(self, group, time, pre_time=0, post_time=0, comment=''):
        """ add trigger to data group

        Parameters
        ----------
        group : int
            group index
        time : float
            trigger time
        pre_time : float
            trigger pre time; default 0
        post_time : float
            trigger post time; default 0
        comment : str
            trigger comment

        """
        gp = self.groups[group]
        trigger, trigger_text = gp['trigger']
        if trigger:
            nr = trigger['trigger_event_nr']
            trigger['trigger_event_nr'] += 1
            trigger['block_len'] += 24
            trigger['trigger_{}_time'.format(nr)] = time
            trigger['trigger_{}_pretime'.format(nr)] = pre_time
            trigger['trigger_{}_posttime'.format(nr)] = post_time
            if trigger_text is None and comment:
                trigger_text = TextBlock.from_text(comment)
                gp['trigger'][1] = trigger_text
        else:
            trigger = TriggerBlock(trigger_event_nr=1,
                                   trigger_0_time=time,
                                   trigger_0_pretime=pre_time,
                                   trigger_0_posttime=post_time)
            if comment:
                trigger_text = TextBlock.from_text(comment)
            else:
                trigger_text = None

            gp['trigger'] = [trigger, trigger_text]

    def append(self, signals, acquisition_info='Python', common_timebase=False):
        """
        Appends a new data group.

        Parameters
        ----------
        signals : list
            list on *Signal* objects
        acquisition_info : str
            acquisition information; default 'Python'
        common_timebase : bool
            flag to hint that the signals have the same timebase

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
        >>> s1 = Signal(samples=s1, timstamps=t, unit='+', name='Positive')
        >>> s2 = Signal(samples=s2, timstamps=t, unit='-', name='Negative')
        >>> s3 = Signal(samples=s3, timstamps=t, unit='flts', name='Floats')
        >>> mdf = MDF3('new.mdf')
        >>> mdf.append([s1, s2, s3], 'created by asammdf v1.1.0')
        >>> # case 2: VTAB conversions from channels inside another file
        >>> mdf1 = MDF3('in.mdf')
        >>> ch1 = mdf1.get("Channel1_VTAB")
        >>> ch2 = mdf1.get("Channel2_VTABR")
        >>> sigs = [ch1, ch2]
        >>> mdf2 = MDF3('out.mdf')
        >>> mdf2.append(sigs, 'created by asammdf v1.1.0')

        """
        dg_cntr = len(self.groups)
        gp = {}
        self.groups.append(gp)

        channel_nr = len(signals)
        if not channel_nr:
            raise MdfException('"append" requires a non-empty list of Signal objects')



        t_ = signals[0].timestamps
        if not common_timebase:
            for s in signals[1:]:
                if not array_equal(s.timestamps, t_):
                    different = True
                    break
            else:
                different = False

            if different:
                times = [s.timestamps for s in signals]
                t = reduce(union1d, times).flatten().astype(float64)
                signals = [s.interp(t) for s in signals]
                times = None
            else:
                t = t_
        else:
            t = t_

        cycles_nr = len(t)

        t_type, t_size = fmt_to_datatype(t.dtype)

        gp['channels'] = gp_channels = []
        gp['channel_conversions'] = gp_conv = []
        gp['channel_extensions'] = gp_source = []
        gp['texts'] = gp_texts = {'channels': [],
                                  'conversion_tab': [],
                                  'channel_group': []}

        #time channel texts
        for _, item in gp_texts.items():
            item.append({})

        gp_texts['channel_group'][-1]['comment_addr'] = TextBlock.from_text(acquisition_info)

        #channels texts
        names = [s.name for s in signals]
        for name in names:
            for _, item in gp['texts'].items():
                item.append({})
            if len(name) >= 32:
                gp_texts['channels'][-1]['long_name_addr'] = TextBlock.from_text(name)

        #conversion for time channel
        kargs = {'conversion_type': CONVERSION_TYPE_NONE,
                 'unit': 's'.encode('latin-1'),
                 'min_phy_value': t[0] if cycles_nr else 0,
                 'max_phy_value': t[-1] if cycles_nr else 0}
        gp_conv.append(ChannelConversion(**kargs))

        # conversions for channels
        if cycles_nr:
            min_max = []
            # compute min and max valkues for all channels
            # for string channels we append (1,0) and use this as a marker (if min>max then channel is string)
            for s in signals:
                if issubdtype(s.samples.dtype, flexible):
                    min_max.append((1,0))
                else:
                    min_max.append((amin(s.samples), amax(s.samples)))
        else:
            min_max = [(0, 0) for s in signals]

        #conversion for channels
        for idx, s in enumerate(signals):
            conv = s.conversion
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
                    kargs['unit'] = s.unit.encode('latin-1')
                elif conv_type == CONVERSION_TYPE_VTABR:
                    kargs = {}
                    kargs['conversion_type'] = CONVERSION_TYPE_VTABR
                    lower = conv['lower']
                    upper = conv['upper']
                    texts = conv['phys']
                    kargs['unit'] = s.unit.encode('latin-1')
                    kargs['ref_param_nr'] = len(upper)

                    for i, (u_, l_, t_) in enumerate(zip(upper, lower, texts)):
                        kargs['lower_{}'.format(i)] = l_
                        kargs['upper_{}'.format(i)] = u_
                        kargs['text_{}'.format(i)] = 0
                        gp_texts['conversion_tab'][-1]['text_{}'.format(i)] = TextBlock.from_text(t_)

                else:
                     kargs = {'conversion_type': CONVERSION_TYPE_NONE,
                              'unit': s.unit.encode('latin-1'),
                              'min_phy_value': min_max[idx][0],
                              'max_phy_value': min_max[idx][1]}
                gp_conv.append(ChannelConversion(**kargs))
            else:
                kargs = {'conversion_type': CONVERSION_TYPE_NONE,
                         'unit': s.unit.encode('latin-1') ,
                         'min_phy_value': min_max[idx][0] if min_max[idx][0]<=min_max[idx][1] else 0,
                         'max_phy_value': min_max[idx][1] if min_max[idx][0]<=min_max[idx][1] else 0}
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
        ch = Channel(**kargs)
        ch.name = 't'
        gp_channels.append(ch)
        self.masters_db[dg_cntr] = 0

        sig_dtypes = [s.samples.dtype for s in signals]
        sig_formats = [fmt_to_datatype(typ) for typ in sig_dtypes]

        #channels
        offset = t_size
        ch_cntr = 1
        for (sigmin, sigmax), (sig_type, sig_size), s in zip(min_max, sig_formats, signals):
            kargs = {'short_name': (s.name[:31] + '\x00').encode('latin-1') if len(s.name) >= 32 else s.name.encode('latin-1'),
                     'channel_type': CHANNEL_TYPE_VALUE,
                     'data_type': sig_type,
                     'lower_limit': sigmin if sigmin<=sigmax else 0,
                     'upper_limit': sigmax if sigmin<=sigmax else 0,
                     'start_offset': offset,
                     'bit_count': sig_size}
            ch = Channel(**kargs)
            ch.name = s.name
            gp_channels.append(ch)
            offset += sig_size

            if s.name in self.channels_db:
                self.channels_db[s.name].append((dg_cntr, ch_cntr))
            else:
                self.channels_db[s.name] = []
                self.channels_db[s.name].append((dg_cntr, ch_cntr))

            ch_cntr += 1

        #channel group
        kargs = {'cycles_nr': cycles_nr,
                 'samples_byte_nr': offset // 8}
        gp['channel_group'] = ChannelGroup(**kargs)
        gp['channel_group']['ch_nr'] = channel_nr + 1

        #data block
        types = [('t', t.dtype),]
        types.extend([(name, typ) for name, typ in zip(names, sig_dtypes)])
        types = dtype(types)

        gp['types'] = types


        parents = {'t': ('t', 0)}
        for name in names:
            parents[name] = name, 0
        gp['parents'] = parents

        arrays = [t, ]
        arrays.extend(s.samples for s in signals)

        samples = fromarrays(arrays, dtype=types)
        block = samples.tostring()

        kargs = {'data': block}
        gp['data_block'] = DataBlock(**kargs)

        #data group
        kargs = {'block_len': DG32_BLOCK_SIZE if self.version in ('3.20', '3.30') else DG31_BLOCK_SIZE}
        gp['data_group'] = DataGroup(**kargs)

        # data group trigger
        gp['trigger'] = [None, None]

    def get(self, name=None, group=None, index=None, raster=None, samples_only=False):
        """Gets channel samples.
        Channel can be specified in two ways:

        * using the first positional argument *name*

            * if there are multiple occurances for this channel then the *group* argument can be used to select a specific group.
            * if there are multiple occurances for this channel and the *group* argument is None then a warning is issued

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
        samples_only : bool
            if *True* return only the channel samples as numpy array; if *False* return a *Signal* object

        Returns
        -------
        res : (numpy.array | Signal)
            returns *Signal* if *samples_only*=*False* (default option), otherwise returns numpy.array

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
            if not name in self.channels_db:
                raise MdfException('Channel "{}" not found'.format(name))
            else:
                if group is None:
                    gp_nr, ch_nr = self.channels_db[name][0]
                    if len(self.channels_db[name]) > 1:
                        warnings.warn('Multiple occurances for channel "{}". Using first occurance from data group {}. Use the "group" argument to select another data group'.format(name, gp_nr))
                else:
                    for gp_nr, ch_nr in self.channels_db[name]:
                        if gp_nr == group:
                            break
                    else:
                        gp_nr, ch_nr = self.channels_db[name][0]
                        warnings.warn('You have selected group "{}" for channel "{}", but this channel was not found in this group. Using first occurance of "{}" from group "{}"'.format(group, name, name, gp_nr))


        grp = self.groups[gp_nr]
        channel = grp['channels'][ch_nr]
        conversion = grp['channel_conversions'][ch_nr]

        try:
            parents, dtypes = grp['parents'], grp['types']
        except:
            grp['parents'], grp['types'] = self._prepare_record(grp)
            parents, dtypes = grp['parents'], grp['types']
        # get data group record
        if not self.load_measured_data:
            data = self._load_group_data(grp)
            record = fromstring(data, dtype=dtypes)
        else:
            try:
                record = grp['record']
            except:
                record = grp['record'] = fromstring(grp['data_block']['data'], dtype=dtypes)

        # check if this is a channel array
        if channel['ch_depend_addr']:
            arrays = [self.get(ch.name, samples_only=True) for ch in channel.dependencies]
            vals = fromarrays(arrays)
            info = None

            if samples_only:
                return vals
            else:
                if conversion:
                    unit = conversion['unit'].decode('latin-1').strip(' \n\t\x00')
                else:
                    unit = ''
                comment = channel['description'].decode('latin-1').strip(' \t\n\x00')

                # get master channel index
                time_ch_nr = self.masters_db[gp_nr]


                time_conv = grp['channel_conversions'][time_ch_nr]
                time_ch = grp['channels'][time_ch_nr]

                t = record[time_ch.name]
                # get timestamps
                time_conv_type = CONVERSION_TYPE_NONE if time_conv is None else time_conv['conversion_type']
                if time_conv_type == CONVERSION_TYPE_LINEAR:
                    time_a = time_conv['a']
                    time_b = time_conv['b']
                    t = t * time_a
                    if time_b:
                        t += time_b
                res = Signal(samples=vals,
                             timestamps=t,
                             unit=unit,
                             name=channel.name,
                             comment=comment)

                if raster and t:
                    tx = linspace(0, t[-1], int(t[-1] / raster))
                    res = res.interp(tx)
                return res
        else:


            # get channel values
            parent, bit_offset = parents[channel.name]
            vals = record[parent]

            if bit_offset:
                vals = vals >> bit_offset
            bits = channel['bit_count']
            if bits % 8:
                vals = vals & ((1<<bits) - 1)

            info = None

            conversion_type = CONVERSION_TYPE_NONE if conversion is None else conversion['conversion_type']

            if conversion_type == CONVERSION_TYPE_NONE:
                # is it a Byte Array?
                if channel['data_type'] == DATA_TYPE_BYTEARRAY:
                    vals = vals.tostring()
                    size = max(bits>>3, 1)
                    cols = size
                    lines = len(vals) // cols

                    vals = frombuffer(vals, dtype=uint8).reshape((lines, cols))

            elif conversion_type == CONVERSION_TYPE_LINEAR:
                a = conversion['a']
                b = conversion['b']
                if (a, b) == (1, 0):
                    size = max(bits>>3, 1)
                    ch_fmt = get_fmt(channel['data_type'], size)
                    if not vals.dtype == ch_fmt:
                        vals = vals.astype(ch_fmt)
                else:
                    vals = vals * a
                    if b:
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

                texts = array([gp['texts']['conversion_tab'][ch_nr].get('text_{}'.format(i), {}).get('text', b'') for i in range(nr)])
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
                vals = evaluate('(P1 * X**2 + P2 * X + P3) / (P4 * X**2 + P5 * X + P6)')

            elif conversion_type == CONVERSION_TYPE_POLY:
                P1 = conversion['P1']
                P2 = conversion['P2']
                P3 = conversion['P3']
                P4 = conversion['P4']
                P5 = conversion['P5']
                P6 = conversion['P6']
                X = values['vals']
                vals = evaluate('(P2 - (P4 * (X - P5 -P6))) / (P3* (X - P5 - P6) - P1)')

            elif conversion_type == CONVERSION_TYPE_FORMULA:
                formula = conversion['formula'].decode('latin-1').strip(' \n\t\x00')
                X1 = values['vals']
                vals = evaluate(formula)

            if samples_only:
                return vals
            else:
                if conversion:
                    unit = conversion['unit'].decode('latin-1').strip(' \n\t\x00')
                else:
                    unit = ''
                comment = channel['description'].decode('latin-1').strip(' \t\n\x00')

                # get master channel index
                time_ch_nr = self.masters_db[gp_nr]

                if time_ch_nr == ch_nr:
                    res = Signal(samples=vals.copy(),
                                 timestamps=vals,
                                 unit=unit,
                                 name=channel.name,
                                 comment=comment)
                else:
                    time_conv = grp['channel_conversions'][time_ch_nr]
                    time_ch = grp['channels'][time_ch_nr]
                    t = record[time_ch.name]
                    # get timestamps
                    time_conv_type = CONVERSION_TYPE_NONE if time_conv is None else time_conv['conversion_type']
                    if time_conv_type == CONVERSION_TYPE_LINEAR:
                        time_a = time_conv['a']
                        time_b = time_conv['b']
                        t = t * time_a
                        if time_b:
                            t += time_b
                    res = Signal(samples=vals,
                                 timestamps=t,
                                 unit=unit,
                                 name=channel.name,
                                 comment=comment,
                                 conversion=info)

                if raster and t:
                    tx = linspace(0, t[-1], int(t[-1] / raster))
                    res = res.interp(tx)
                return res

    def iter_get_triggers(self):
        """ generator that yields triggers

        Returns
        -------
        trigger_info : dict
            trigger information with the following keys:

                * comment : trigger comment
                * time : trigger time
                * pre_time : trigger pre time
                * post_time : trigger post time
                * index : trigger index
                * group : data group index of trigger
        """
        for i, gp in enumerate(self.groups):
            trigger, trigger_text = gp['trigger']
            if trigger:
                if trigger_text:
                    comment = trigger_text.text_str
                else:
                    comment = ''

                for j in range(trigger['trigger_events_nr']):
                    trigger_info = {'comment': comment,
                                    'index' : j,
                                    'group': i,
                                    'time' : trigger['trigger_{}_time'.format(j)],
                                    'pre_time' : trigger['trigger_{}_pretime'.format(j)],
                                    'post_time' : trigger['trigger_{}_posttime'.format(j)]}
                    yield trigger_info

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

    def remove(self, group=None, name=None):
        """Remove data group. Use *group* or *name* keyword arguments to identify the group's index. *group* has priority

        Parameters
        ----------
        name : string
            name of the channel inside the data group to be removed
        group : int
            data group index to be removed

        Examples
        --------
        >>> mdf = MDF3('test.mdf')
        >>> mdf.remove(group=3)
        >>> mdf.remove(name='VehicleSpeed')

        """
        if self.load_measured_data == False:
            warnings.warn("Can't remove group if load_measurement_data option is False")
            return

        if group:
            if 0 <= group <= len(self.groups):
                idx = group
            else:
                print('Group index "{}" not in valid range[0..{}]'.format(group, len(self.groups)))
                return
        elif name:
            if name in self.channels_db:
                idx = self.channels_db[name][1]
            else:
                print('Channel name "{}" not found in the measurement'.format(name))
                return
        else:
            print('Must specify a valid group or name argument')
            return
        self.groups.pop(idx)

    def save(self, dst=None):
        """Save MDF to *dst*. If *dst* is *None* the original file is overwritten

        """

        if self.file_history is None:
            self.file_history = TextBlock.from_text('''<FHcomment>
<TX>created</TX>
<tool_id>PythonMDFEditor</tool_id>
<tool_vendor> </tool_vendor>
<tool_version>1.0</tool_version>
</FHcomment>''')
        else:
            text = self.file_history['text'] + '\n{}: updated byt Python script'.format(time.asctime()).encode('latin-1')
            self.file_history = TextBlock.from_text(text)

        if self.name is None and dst is None:
            print('New MDF created without a name and no destination file name specified for save')
            return
        dst = dst if dst else self.name

        if PYVERSION == 2:
            self._save_py2(dst)
        else:
            self._save_py3(dst)

    def _save_py2(self, dst):
        with open(dst, 'wb') as dst:
            #store unique texts and their addresses
            defined_texts = {}
            address = 0

            write = dst.write
            tell = dst.tell

            write(bytes(self.identification))
            address = tell()
            write(bytes(self.header))
            address = tell()

            self.file_history.address = address
            write(bytes(self.file_history))
            address = tell()

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
                                write(bytes(my_dict[key]))
                                address = tell()

                # ChannelConversions
                cc = gp['channel_conversions']
                for i, conv in enumerate(cc):
                    if conv:
                        conv.address = address
                        if conv['conversion_type'] == CONVERSION_TYPE_VTABR:
                            for key, item in gp_texts['conversion_tab'][i].items():
                                conv[key] = item.address

                        write(bytes(conv))
                        address = tell()

                # Channel Extension
                cs = gp['channel_extensions']
                for source in cs:
                    if source:
                        source.address = address
                        write(bytes(source))
                        address = tell()

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
                    write(bytes(channel))
                next_channel['next_ch_addr'] = 0
                write(bytes(next_channel))
                address = tell()

                # ChannelGroup
                cg = gp['channel_group']
                cg.address = address

                cg['first_ch_addr'] = gp['channels'][0].address
                cg['next_cg_addr'] = 0
                if 'comment_addr' in gp['texts']['channel_group'][0]:
                    cg['comment_addr'] = gp_texts['channel_group'][0]['comment_addr'].address
                write(bytes(cg))
                address = tell()

                # TriggerBLock
                trigger, trigger_text = gp['trigger']
                if trigger:
                    if trigger_text:
                        trigger_text.address = address
                        write(bytes(trigger_text))
                        address = tell()
                        trigger['comment_addr'] = trigger_text.address
                    else:
                        trigger['comment_addr'] = 0

                    trigger.address = address
                    write(bytes(trigger))
                    address = tell()

                # DataBlock
                if self.load_measured_data == False:
                    # check if there are appended blocks
                    if gp['data_block']:
                        data = gp['data_block']['data']
                    else:
                        data = self._load_group_data(gp)
                    data_block_address = address
                    write(data)
                    address = tell()

                else:
                    db = gp['data_block']
                    db.address = address
                    data_block_address = address
                    write(bytes(db))
                    address = tell()
                gp['data_group']['data_block_addr'] = data_block_address

            # DataGroup and DataBlock
            for i, gp in enumerate(self.groups):
                dg = gp['data_group']
                dg.address = address
                address += dg['block_len']
                dg['first_cg_addr'] = gp['channel_group'].address
                dg['trigger_addr'] = gp['trigger'][0].address if gp['trigger'][0] else 0

            for i, dg in enumerate(self.groups[:-1]):
                dg['data_group']['next_dg_addr'] = self.groups[i+1]['data_group'].address
            self.groups[-1]['data_group']['next_dg_addr'] = 0

            for dg in self.groups:
                write(bytes(dg['data_group']))

            if self.groups:
                self.header['first_dg_addr'] = self.groups[0]['data_group'].address
                self.header['dg_nr'] = len(self.groups)
                self.header['comment_addr'] = self.file_history.address
                self.header['program_addr'] = 0
            dst.seek(0, SEEK_START)
            write(bytes(self.identification))
            write(bytes(self.header))

    def _save_py3(self, dst):
        with open(dst, 'wb') as dst:
            #store unique texts and their addresses
            defined_texts = {}
            address = 0

            write = dst.write
            tell = dst.tell

            address += write(bytes(self.identification))
            address += write(bytes(self.header))

            self.file_history.address = address
            address += write(bytes(self.file_history))

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
                                address += write(bytes(my_dict[key]))

                # ChannelConversions
                cc = gp['channel_conversions']
                for i, conv in enumerate(cc):
                    if conv:
                        conv.address = address
                        if conv['conversion_type'] == CONVERSION_TYPE_VTABR:
                            for key, item in gp_texts['conversion_tab'][i].items():
                                conv[key] = item.address

                        address += write(bytes(conv))

                # Channel Extension
                cs = gp['channel_extensions']
                for source in cs:
                    if source:
                        source.address = address
                        address += write(bytes(source))

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
                    write(bytes(channel))
                next_channel['next_ch_addr'] = 0
                write(bytes(next_channel))

                # ChannelGroup
                cg = gp['channel_group']
                cg.address = address

                cg['first_ch_addr'] = gp['channels'][0].address
                cg['next_cg_addr'] = 0
                if 'comment_addr' in gp['texts']['channel_group'][0]:
                    cg['comment_addr'] = gp_texts['channel_group'][0]['comment_addr'].address
                address += write(bytes(cg))

                # TriggerBLock
                trigger, trigger_text = gp['trigger']
                if trigger:
                    if trigger_text:
                        trigger_text.address = address
                        address += write(bytes(trigger_text))
                        trigger['comment_addr'] = trigger_text.address
                    else:
                        trigger['comment_addr'] = 0

                    trigger.address = address
                    address += write(bytes(trigger))

                # DataBlock
                if self.load_measured_data == False:
                    # check if there are appended blocks
                    if gp['data_block']:
                        data = gp['data_block']['data']
                    else:
                        data = self._load_group_data(gp)
                    data_block_address = address
                    address += write(data)
                else:
                    db = gp['data_block']
                    db.address = address
                    data_block_address = address
                    address += write(bytes(db))
                    address = tell()
                gp['data_group']['data_block_addr'] = data_block_address

            # DataGroup
            for gp in self.groups:
                dg = gp['data_group']
                dg.address = address
                address += dg['block_len']
                dg['first_cg_addr'] = gp['channel_group'].address
                dg['trigger_addr'] = gp['trigger'][0].address if gp['trigger'][0] else 0

            for i, dg in enumerate(self.groups[:-1]):
                dg['data_group']['next_dg_addr'] = self.groups[i+1]['data_group'].address
            self.groups[-1]['data_group']['next_dg_addr'] = 0

            for dg in self.groups:
                write(bytes(dg['data_group']))

            if self.groups:
                self.header['first_dg_addr'] = self.groups[0]['data_group'].address
                self.header['dg_nr'] = len(self.groups)
                self.header['comment_addr'] = self.file_history.address
                self.header['program_addr'] = 0
            dst.seek(0, SEEK_START)
            write(bytes(self.identification))
            write(bytes(self.header))


if __name__ == '__main__':
    pass

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
from tempfile import TemporaryFile
from itertools import product

from numpy import (interp, linspace, dtype, amin, amax, array_equal,
                   array, searchsorted, log, exp, clip, union1d, float64,
                   uint8, frombuffer, issubdtype, flexible, arange, recarray,
                   column_stack)
from numpy.core.records import fromstring, fromarrays
from numpy.core.defchararray import decode, encode
from numexpr import evaluate

from .utils import MdfException, get_fmt, pair, fmt_to_datatype
from .signal import Signal
from .v3constants import *
from .v3blocks import (Channel, ChannelConversion, ChannelDependency,
                       ChannelExtension, ChannelGroup, DataBlock, DataGroup,
                       FileIdentificationBlock, HeaderBlock, ProgramBlock,
                       SampleReduction, TextBlock, TriggerBlock)


if PYVERSION == 2:
    from .utils import bytes


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
    version : str
        mdf version
    channels_db : dict
        used for fast channel access by name; for each name key the value is a list of (group index, channel index) tuples
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

        # used when appending to MDF object created with load_measured_data=False
        self._tempfile = None

        if name:
            self._read()
        else:
            self.identification = FileIdentificationBlock(version=version)
            self.version = version
            self.header = HeaderBlock(version=self.version)

    def _load_group_data(self, group):
        """ get group's data block bytes"""
        if self.load_measured_data == False:
            # could be an appended group
            # for now appended groups keep the measured data in the memory.
            # the plan is to use a temp file for appended groups, to keep the
            # memory usage low.
            if group['data_location'] == LOCATION_ORIGINAL_FILE:
                # this is a group from the source file
                # so fetch the measured data from it
                with open(self.name, 'rb') as file_stream:
                    # go to the first data block of the current data group
                    dat_addr = group['data_group']['data_block_addr']

                    if group.get('sorted', True):
                        read_size = group['size']
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
            elif group['data_location'] == LOCATION_TEMPORARY_FILE:
                read_size = group['size']
                dat_addr = group['data_group']['data_block_addr']
                self._tempfile.seek(dat_addr, SEEK_START)
                data = self._tempfile.read(read_size)
        else:
            data = group['data_block']['data']
        return data

    def _prepare_record(self, group):
        """ compute record dtype and parents dict for this group

        Parameters
        ----------
        group : dict
            MDF group dict

        Returns
        -------
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
        group_channels = set()

        # the channels are first sorted ascending (see __lt__ method of Channel class):
        # a channel with lower start offset is smaller, when two channels have
        # the same start offset the one with higer bit size is considered smaller.
        # The reason is that when the numpy record is built and there are overlapping
        # channels, the parent fields should be bigger (bit size) than the embedded
        # channels. For each channel the parent dict will have a (parent name, bit offset) pair:
        # the channel value is computed using the values from the parent field,
        # and the bit offset, which is the channel's bit offset within the parent bytes.
        # This means all parents will have themselves as parent, and bit offset of 0.
        # Gaps in the records are also considered. Non standard integers size is
        # adjusted to the first higher standard integer size (eq. uint of 28bits will
        # be adjusted to 32bits)

        for original_index, new_ch in sorted(enumerate(grp['channels']), key=lambda i: i[1]):
            # channels with channel dependencies are skipped from the numpy record
            if new_ch['ch_depend_addr']:
                continue

            start_offset = new_ch['start_offset']
            bit_offset = start_offset % 8
            data_type = new_ch['data_type']
            bit_count = new_ch['bit_count']
            name = new_ch.name
            if PYVERSION == 2:
                name = str(new_ch.name)

            # handle multiple occurance of same channel name
            i = 0
            new_name = name
            while new_name in group_channels:
                new_name = "{}_{}".format(name, i)
                i += 1
            group_channels.add(name)

            if start_offset >= next_byte_aligned_position:
                parent_start_offset = (start_offset >> 3 ) << 3
                parents[original_index] = name, bit_offset

                # check if there are byte gaps in the record
                gap = (parent_start_offset - next_byte_aligned_position) >> 3
                if gap:
                    types.append( ('', 'a{}'.format(gap)) )

                # adjust size to 1, 2, 4 or 8 bytes for nonstandard integers
                size = bit_offset + bit_count
                if data_type == DATA_TYPE_STRING:
                    next_byte_aligned_position = parent_start_offset + size
                    size = size >> 3
                    types.append( (name, get_fmt(data_type, size)) )

                elif data_type == DATA_TYPE_BYTEARRAY:
                    size = size >> 3
                    types.append( (name, 'u1', (size, 1)) )

                else:
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
                parents[original_index] = current_parent, start_offset - parent_start_offset

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

            self.version = self.identification['version_str'].decode('latin-1').strip(' \n\t\x00')

            self.file_history = TextBlock(address=self.header['comment_addr'], file_stream=file_stream)

            # this will hold mapping from channel address to Channel object
            # needed for linking dependecy blocks to refernced channels after the file is loaded
            ch_map = {}

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
                    if trigger['text_addr']:
                        trigger_text = TextBlock(address=trigger['text_addr'], file_stream=file_stream)
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
                    grp['channel_dependencies'] = []

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

                        # check if it has channel dependencies
                        if new_ch['ch_depend_addr']:
                            grp['channel_dependencies'].append(ChannelDependency(address=new_ch['ch_depend_addr'], file_stream=file_stream))
                        else:
                            grp['channel_dependencies'].append(None)

                        # update channel map
                        ch_map[ch_addr] = (new_ch, grp)

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
                            new_ch.name = ch_texts['long_name_addr']['text'].decode('latin-1').strip(' \n\t\x00')
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
                    # this is an unsorted file since there are multiple channel groups
                    # within a data group
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
                    grp['size'] = size = (grp['channel_group']['samples_byte_nr'] + record_id_nr) * grp['channel_group']['cycles_nr']

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
                        grp['data_location'] = LOCATION_MEMORY
                        kargs = {'data': data}
                        grp['data_block'] = DataBlock(**kargs)

                    else:
                        # agregate data for each record ID in the cg_data dict
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
                            grp['data_location'] = LOCATION_MEMORY
                            kargs = {}
                            kargs['data'] = b''.join(cg_data[grp['channel_group']['record_id']])
                            grp['channel_group']['record_id'] = 1
                            grp['data_block'] = DataBlock(**kargs)
                else:
                    for grp in new_groups:
                        grp['data_location'] = LOCATION_ORIGINAL_FILE

                self.groups.extend(new_groups)

                # go to next data group
                dg_addr = gp['next_dg_addr']

            # once the file has been loaded update the channel depency refenreces
            for grp in self.groups:
                for dependency_block in grp['channel_dependencies']:
                    if dependency_block:
                        for i in range(dependency_block['sd_nr']):
                            ref_channel_addr = dependency_block['ch_{}'.format(i)]
                            dependency_block.referenced_channels.append(ch_map[ref_channel_addr])

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
                trigger_text = TextBlock(text=comment)
                gp['trigger'][1] = trigger_text
        else:
            trigger = TriggerBlock(trigger_event_nr=1,
                                   trigger_0_time=time,
                                   trigger_0_pretime=pre_time,
                                   trigger_0_posttime=post_time)
            if comment:
                trigger_text = TextBlock(text=comment)
            else:
                trigger_text = None

            gp['trigger'] = [trigger, trigger_text]

    def append(self, signals, acquisition_info='Python', common_timebase=False):
        """
        Appends a new data group.

        For channel depencies type Signals, the *samples* attribute must be a numpy.recarray

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
        if not signals:
            raise MdfException('"append" requires a non-empty list of Signal objects')

        # check if the signals have a common timebase
        # if not interpolate the signals using the union of all timbases
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

        # split for signals that come from a channel dependency
        # (this means the samples will be of type np.composed)
        # from the regular one dimensional signals.
        # The regular signals will be first added to the group.
        # The composed signals will be saved along side the fields, which will
        # be saved as new signals.
        simple_signals = [sig for sig in signals if len(sig.samples.shape) <= 1 and sig.samples.dtype.names is None]
        composed_signals = [sig for sig in signals if len(sig.samples.shape) > 1 or sig.samples.dtype.names]

        # mdf version 4 structure channels and CANopen types will be saved to new channel groups
        new_groups_signals = [sig for sig in composed_signals if sig.samples.dtype.names and sig.samples.dtype.names[0] != sig.name]
        composed_signals = [sig for sig in composed_signals if not sig.samples.dtype.names or sig.samples.dtype.names[0] == sig.name]

        if simple_signals or composed_signals:
            dg_cntr = len(self.groups)

            gp = {}
            gp['channels'] = gp_channels = []
            gp['channel_conversions'] = gp_conv = []
            gp['channel_extensions'] = gp_source = []
            gp['channel_dependencies'] = gp_dep = []
            gp['texts'] = gp_texts = {'channels': [],
                                      'conversion_tab': [],
                                      'channel_group': []}
            self.groups.append(gp)

            cycles_nr = len(t)

            # setup all blocks related to the time master channel
            t_type, t_size = fmt_to_datatype(t.dtype)

            # time channel texts
            for _, item in gp_texts.items():
                item.append({})

            gp_texts['channel_group'][-1]['comment_addr'] = TextBlock(text=acquisition_info)

            #conversion for time channel
            kargs = {'conversion_type': CONVERSION_TYPE_NONE,
                     'unit': 's'.encode('latin-1'),
                     'min_phy_value': t[0] if cycles_nr else 0,
                     'max_phy_value': t[-1] if cycles_nr else 0}
            gp_conv.append(ChannelConversion(**kargs))

            #source for time
            kargs = {'module_nr': 0,
                     'module_address': 0,
                     'type': SOURCE_ECU,
                     'description': 'Channel inserted by Python Script'.encode('latin-1')}
            gp_source.append(ChannelExtension(**kargs))

            #time channel
            kargs = {'short_name': 't'.encode('latin-1'),
                     'channel_type': CHANNEL_TYPE_MASTER,
                     'data_type': t_type,
                     'start_offset': 0,
                     'min_raw_value' : t[0] if cycles_nr else 0,
                     'max_raw_value' : t[-1] if cycles_nr else 0,
                     'bit_count': t_size}
            ch = Channel(**kargs)
            ch.name = name = 't'
            gp_channels.append(ch)

            ch_cntr = 0
            if not name in self.channels_db:
                self.channels_db[name] = []
            self.channels_db[name].append((dg_cntr, ch_cntr))
            self.masters_db[dg_cntr] = 0

            # time channel doesn't have channel dependencies
            gp_dep.append(None)

            # prepare start bit offset and channel counter for the other channels
            offset = t_size
            ch_cntr += 1

            # arrays will hold the samples for all channels
            # types holds the (channel name, numpy dtype) pairs for all channels
            # formats holds the (ASAM channel data type, bit size) for all channels
            arrays = [t, ]
            types = [('t', t.dtype),]
            formats = [fmt_to_datatype(t.dtype), ]

            # data group record parents
            parents = {0: ('t', 0)}

            # first add the signals in the simple signal list
            if simple_signals:
                # channels texts
                names = [s.name for s in simple_signals]
                for name in names:
                    for _, item in gp['texts'].items():
                        item.append({})
                    if len(name) >= 32:
                        gp_texts['channels'][-1]['long_name_addr'] = TextBlock(text=name)

                sig_dtypes = []
                sig_formats = []
                for i, sig in enumerate(simple_signals):
                    sig_dtypes.append(sig.samples.dtype)
                    sig_formats.append(fmt_to_datatype(sig.samples.dtype))

                # conversions for channels
                if cycles_nr:
                    min_max = []
                    # compute min and max values for all channels
                    # for string channels we append (1,0) and use this as a marker (if min>max then channel is string)
                    for s in simple_signals:
                        if issubdtype(s.samples.dtype, flexible):
                            min_max.append((1,0))
                        else:
                            min_max.append((amin(s.samples), amax(s.samples)))
                else:
                    min_max = [(0, 0) for s in simple_signals]

                #conversion for channels
                for idx, s in enumerate(simple_signals):
                    info = s.info
                    if info:
                        if 'raw' in info:
                            kargs = {}
                            kargs['conversion_type'] = CONVERSION_TYPE_VTAB
                            raw = info['raw']
                            phys = info['phys']
                            for i, (r_, p_) in enumerate(zip(raw, phys)):
                                kargs['text_{}'.format(i)] = p_[:31] + b'\x00'
                                kargs['param_val_{}'.format(i)] = r_
                            kargs['ref_param_nr'] = len(raw)
                            kargs['unit'] = s.unit.encode('latin-1')
                        elif 'lower' in info:
                            kargs = {}
                            kargs['conversion_type'] = CONVERSION_TYPE_VTABR
                            lower = info['lower']
                            upper = info['upper']
                            texts = info['phys']
                            kargs['unit'] = s.unit.encode('latin-1')
                            kargs['ref_param_nr'] = len(upper)

                            for i, (u_, l_, t_) in enumerate(zip(upper, lower, texts)):
                                kargs['lower_{}'.format(i)] = l_
                                kargs['upper_{}'.format(i)] = u_
                                kargs['text_{}'.format(i)] = 0
                                gp_texts['conversion_tab'][-1]['text_{}'.format(i)] = TextBlock(text=t_)

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

                #source for channels
                for _ in simple_signals:
                    kargs = {'module_nr': 0,
                             'module_address': 0,
                             'type': SOURCE_ECU,
                             'description': 'Channel inserted by Python Script'.encode('latin-1')}
                    gp_source.append(ChannelExtension(**kargs))

                #channels
                for (sigmin, sigmax), (sig_type, sig_size), s in zip(min_max, sig_formats, simple_signals):
                    # compute additional byte offset for large records size
                    if offset > MAX_UINT16:
                        additional_byte_offset = (offset - MAX_UINT16 ) >> 3
                        start_bit_offset = offset - additional_byte_offset << 3
                    else:
                        start_bit_offset = offset
                        additional_byte_offset = 0
                    kargs = {'short_name': (s.name[:31] + '\x00').encode('latin-1') if len(s.name) >= 32 else s.name.encode('latin-1'),
                             'channel_type': CHANNEL_TYPE_VALUE,
                             'data_type': sig_type,
                             'min_raw_value': sigmin if sigmin <= sigmax else 0,
                             'max_raw_value': sigmax if sigmin <= sigmax else 0,
                             'start_offset': start_bit_offset,
                             'bit_count': sig_size,
                             'aditional_byte_offset' : additional_byte_offset,
                             'description': s.comment.encode('latin-1')}
                    if s.comment:
                        kargs['description'] = (s.comment[:127] + '\x00').encode('latin-1') if len(s.comment) >= 128 else s.comment.encode('latin-1')
                    ch = Channel(**kargs)

                    name = s.name

                    ch.name = name
                    gp_channels.append(ch)
                    offset += sig_size

                    if not name in self.channels_db:
                        self.channels_db[name] = []
                    self.channels_db[name].append((dg_cntr, ch_cntr))

                    # update the parents as well
                    parents[ch_cntr] = name, 0

                    ch_cntr += 1

                # simple channels don't have channel dependencies
                for _ in simple_signals:
                    gp_dep.append(None)

                # extend arrays, types and formats with data related to the simple signals
                arrays.extend(s.samples for s in simple_signals)
                if PYVERSION == 3:
                    types.extend([(name, typ) for name, typ in zip(names, sig_dtypes)])
                else:
                    types.extend([(str(name), typ) for name, typ in zip(names, sig_dtypes)])
                formats.extend(sig_formats)

            # second, add the composed signals
            for sig in composed_signals:
                names = sig.samples.dtype.names
                name = sig.name

                if names:
                    new_names = []
                    signals = []
                    samples = sig.samples[names[0]]

                    shape = samples.shape[1:]
                    dims = [list(range(size)) for size in shape]

                    for indexes in product(*dims):
                        subarray = samples
                        for idx in indexes:
                            subarray = subarray[:, idx]
                        signals.append(subarray)

                        new_names.append('{}{}'.format(name, ''.join('[{}]'.format(idx) for idx in indexes)))

                    # add channel dependency block for composed parent channel
                    sd_nr = len(signals)
                    kargs = {'sd_nr': sd_nr}
                    for i, dim in enumerate(shape[::-1]):
                        kargs['dim_{}'.format(i)] = dim
                    parent_dep = ChannelDependency(**kargs)
                    gp_dep.append(parent_dep)

                    signals.extend([sig.samples[name] for name in names[1:]])

                    # numpy dtype does not allow for reapeating names so we must handle name conflincts
                    dtype_fields = [t[0] for t in types]
                    # if the name already exist in the dtype fiels then
                    # compute a new name "name_xx", by incrementing the index
                    # until a valid new name is found
                    for name in names[1:]:
                        i = 0
                        new_name = name
                        while new_name in dtype_fields:
                            new_name = "{}_{}".format(name, i)
                            i += 1
                        new_names.append(new_name)

                    names = new_names

                else:
                    names = []
                    signals = []

                    shape = sig.samples.shape[1:]
                    dims = [list(range(size)) for size in shape]

                    for indexes in product(*dims):
                        subarray = sig.samples
                        for idx in indexes:
                            subarray = subarray[:, idx]
                        signals.append(subarray)

                        names.append('{}{}'.format(name, ''.join('[{}]'.format(idx) for idx in indexes)))

                    # add channel dependency block for composed parent channel
                    sd_nr = len(signals)
                    kargs = {'sd_nr': sd_nr}
                    for i, dim in enumerate(shape[::-1]):
                        kargs['dim_{}'.format(i)] = dim
                    parent_dep = ChannelDependency(**kargs)
                    gp_dep.append(parent_dep)


                # add composed parent signal texts
                name = sig.name
                for _, item in gp['texts'].items():
                    item.append({})
                if len(name) >= 32:
                    gp_texts['channels'][-1]['long_name_addr'] = TextBlock(text=name)
                # add components texts
                for name in names:
                    for _, item in gp['texts'].items():
                        item.append({})
                    if len(name) >= 32:
                        gp_texts['channels'][-1]['long_name_addr'] = TextBlock(text=name)

                # composed parent has no conversion
                gp_conv.append(None)
                # add components conversions
                min_max = []
                if cycles_nr:
                    for s in signals:
                        min_max.append( (amin(s), amax(s)) )
                else:
                    for s in signals:
                        min_max = [(0, 0)]
                for i, s in enumerate(signals):
                    kargs = {'conversion_type': CONVERSION_TYPE_NONE,
                             'unit': b'',
                             'min_phy_value': min_max[i][0],
                             'max_phy_value': min_max[i][1]}
                    gp_conv.append(ChannelConversion(**kargs))

                # add parent and components sources
                kargs = {'module_nr': 0,
                         'module_address': 0,
                         'type': SOURCE_ECU,
                         'description': 'Channel inserted by Python Script'.encode('latin-1')}
                gp_source.append(ChannelExtension(**kargs))
                for _ in signals:
                    gp_source.append(ChannelExtension(**kargs))

                new_sig_dtypes = [s.dtype for s in signals]
                new_sig_formats = [fmt_to_datatype(typ) for typ in new_sig_dtypes]
                shapes = [s.shape[1:] for s in signals]

                # extend arrays, types and formasts with data from current composed
                arrays.extend(signals)
                if PYVERSION == 3:
                    types.extend([(name, dtype_, shape) for name, shape, dtype_ in zip(names, shapes, new_sig_dtypes)])
                else:
                    types.extend([(str(name), dtype_, shape) for name, shape, dtype_ in zip(names, shapes, new_sig_dtypes)])
                formats.extend(new_sig_formats)

                # components do not have channel dependencies
                for _ in signals:
                    gp_dep.append(None)

                # add parent channel
                if offset > MAX_UINT16:
                    additional_byte_offset = (offset - MAX_UINT16 ) >> 3
                    start_bit_offset = offset - additional_byte_offset << 3
                else:
                    start_bit_offset = offset
                    additional_byte_offset = 0
                kargs = {'short_name': (sig.name[:31] + '\x00').encode('latin-1') if len(sig.name) >= 32 else sig.name.encode('latin-1'),
                         'channel_type': CHANNEL_TYPE_VALUE,
                         'data_type': new_sig_formats[0][0],
                         'min_raw_value': 0,
                         'max_raw_value': 0,
                         'start_offset': start_bit_offset,
                         'bit_count': new_sig_formats[0][1],
                         'aditional_byte_offset' : additional_byte_offset,
                         'description': sig.comment.encode('latin-1')}
                ch = Channel(**kargs)
                ch.name = name = sig.name
                gp_channels.append(ch)

                # update channel database with racaaray parent indexes
                if not name in self.channels_db:
                    self.channels_db[name] = []
                self.channels_db[name].append((dg_cntr, ch_cntr))

                ch_cntr += 1

                # add components channels
                for i, ((sigmin, sigmax), (sig_type, sig_size), shape, name) in enumerate(zip(min_max, new_sig_formats, shapes, names)):
                    if offset > MAX_UINT16:
                        additional_byte_offset = (offset - MAX_UINT16 ) >> 3
                        start_bit_offset = offset - additional_byte_offset << 3
                    else:
                        start_bit_offset = offset
                        additional_byte_offset = 0

                    size = sig_size
                    for dim in shape:
                        size *= dim

                    kargs = {'short_name': (name[:31] + '\x00').encode('latin-1') if len(name) >= 32 else name.encode('latin-1'),
                             'channel_type': CHANNEL_TYPE_VALUE,
                             'data_type': sig_type,
                             'min_raw_value': sigmin if sigmin<=sigmax else 0,
                             'max_raw_value': sigmax if sigmin<=sigmax else 0,
                             'start_offset': start_bit_offset,
                             'bit_count': sig_size,
                             'aditional_byte_offset' : additional_byte_offset}
                    ch = Channel(**kargs)
                    ch.name = name
                    gp_channels.append(ch)
                    offset += size

                    if i < sd_nr:
                        parent_dep.referenced_channels.append((ch, gp))
                    else:
                        ch['description'] = 'axis {} for channel {}'.format(name, sig.name).encode('latin-1')

                    # update channel database with component indexes
                    if not name in self.channels_db:
                        self.channels_db[name] = []
                    self.channels_db[name].append((dg_cntr, ch_cntr))

                    # also update record parents
                    parents[ch_cntr] = name, 0

                    ch_cntr += 1

            #channel group
            kargs = {'cycles_nr': cycles_nr,
                     'samples_byte_nr': offset >> 3}
            gp['channel_group'] = ChannelGroup(**kargs)
            gp['channel_group']['ch_nr'] = ch_cntr
            gp['size'] = cycles_nr * (offset >> 3)

            #data group
            kargs = {'block_len': DG32_BLOCK_SIZE if self.version in ('3.20', '3.30') else DG31_BLOCK_SIZE}
            gp['data_group'] = DataGroup(**kargs)

            #data block

            types = dtype(types)

            gp['types'] = types
            gp['parents'] = parents

            samples = fromarrays(arrays, dtype=types)
            block = samples.tostring()

            if self.load_measured_data:
                gp['data_location'] = LOCATION_MEMORY
                kargs = {'data': block}
                gp['data_block'] = DataBlock(**kargs)
            else:
                gp['data_location'] = LOCATION_TEMPORARY_FILE
                if self._tempfile is None:
                    self._tempfile = TemporaryFile()
                self._tempfile.seek(0, SEEK_END)
                data_address = self._tempfile.tell()
                gp['data_group']['data_block_addr'] = data_address
                self._tempfile.write(block)

            # data group trigger
            gp['trigger'] = [None, None]

        for sig in new_groups_signals:
            dg_cntr = len(self.groups)
            gp = {}
            gp['channels'] = gp_channels = []
            gp['channel_conversions'] = gp_conv = []
            gp['channel_extensions'] = gp_source = []
            gp['channel_dependencies'] = gp_dep = []
            gp['texts'] = gp_texts = {'channels': [],
                                      'conversion_tab': [],
                                      'channel_group': []}
            self.groups.append(gp)

            cycles_nr = len(t)

            # setup all blocks related to the time master channel
            t_type, t_size = fmt_to_datatype(t.dtype)

            # time channel texts
            for _, item in gp_texts.items():
                item.append({})

            names = sig.samples.dtype.names
            if names == ('ms', 'days'):
                gp_texts['channel_group'][-1]['comment_addr'] = TextBlock(text='From mdf version 4 CANopen Time channel')
            elif names == ('ms', 'min', 'hour', 'day', 'month', 'year', 'summer_time', 'day_of_week'):
                gp_texts['channel_group'][-1]['comment_addr'] = TextBlock(text='From mdf version 4 CANopen Date channel')
            else:
                gp_texts['channel_group'][-1]['comment_addr'] = TextBlock(text='From mdf version 4 structure channel composition')

            #conversion for time channel
            kargs = {'conversion_type': CONVERSION_TYPE_NONE,
                     'unit': 's'.encode('latin-1'),
                     'min_phy_value': t[0] if cycles_nr else 0,
                     'max_phy_value': t[-1] if cycles_nr else 0}
            gp_conv.append(ChannelConversion(**kargs))

            #source for time
            kargs = {'module_nr': 0,
                     'module_address': 0,
                     'type': SOURCE_ECU,
                     'description': 'Channel inserted by Python Script'.encode('latin-1')}
            gp_source.append(ChannelExtension(**kargs))

            # time channel
            kargs = {'short_name': 't'.encode('latin-1'),
                     'channel_type': CHANNEL_TYPE_MASTER,
                     'data_type': t_type,
                     'start_offset': 0,
                     'min_raw_value' : t[0] if cycles_nr else 0,
                     'max_raw_value' : t[-1] if cycles_nr else 0,
                     'bit_count': t_size}
            ch = Channel(**kargs)
            ch.name = name = 't'
            gp_channels.append(ch)

            ch_cntr = 0
            if not name in self.channels_db:
                self.channels_db[name] = []
            self.channels_db[name].append((dg_cntr, ch_cntr))
            self.masters_db[dg_cntr] = 0

            # time channel doesn't have channel dependencies
            gp_dep.append(None)

            # prepare start bit offset and channel counter for the other channels
            offset = t_size
            ch_cntr += 1

            # arrays will hold the samples for all channels
            # types holds the (channel name, numpy dtype) pairs for all channels
            # formats holds the (ASAM channel data type, bit size) for all channels
            arrays = [t, ]
            types = [('t', t.dtype),]
            formats = [fmt_to_datatype(t.dtype), ]

            # data group record parents
            parents = {0: ('t', 0)}

            for name in names:
                vals = sig.samples[name]
                arrays.append(vals)
                types.append((name, vals.dtype))

                sig_type, sig_size = fmt_to_datatype(vals.dtype)

                gp_dep.append(None)

                # add signal texts
                for _, item in gp['texts'].items():
                    item.append({})
                if len(name) >= 32:
                    gp_texts['channels'][-1]['long_name_addr'] = TextBlock(text=name)

                min_val, max_val = amin(vals), amax(vals)

                kargs = {'conversion_type': CONVERSION_TYPE_NONE,
                         'unit': b'',
                         'min_phy_value': min_val,
                         'max_phy_value': max_val}
                gp_conv.append(ChannelConversion(**kargs))

                kargs = {'module_nr': 0,
                         'module_address': 0,
                         'type': SOURCE_ECU,
                         'description': 'Channel inserted by Python Script'.encode('latin-1')}
                gp_source.append(ChannelExtension(**kargs))

                if offset > MAX_UINT16:
                    additional_byte_offset = (offset - MAX_UINT16 ) >> 3
                    start_bit_offset = offset - additional_byte_offset << 3
                else:
                    start_bit_offset = offset
                    additional_byte_offset = 0

                kargs = {'short_name': (name[:31] + '\x00').encode('latin-1') if len(name) >= 32 else name.encode('latin-1'),
                         'channel_type': CHANNEL_TYPE_VALUE,
                         'data_type': sig_type,
                         'min_raw_value': min_val,
                         'max_raw_value': max_val,
                         'start_offset': start_bit_offset,
                         'bit_count': sig_size,
                         'aditional_byte_offset' : additional_byte_offset}
                ch = Channel(**kargs)
                ch.name = name
                gp_channels.append(ch)
                offset += sig_size

#                ch['description'] = 'axis {} for channel {}'.format(name, sig.name).encode('latin-1')

                # update channel database with component indexes
                if not name in self.channels_db:
                    self.channels_db[name] = []
                self.channels_db[name].append((dg_cntr, ch_cntr))

                # also update record parents
                parents[ch_cntr] = name, 0

                ch_cntr += 1

            # channel group
            kargs = {'cycles_nr': cycles_nr,
                     'samples_byte_nr': offset >> 3}
            gp['channel_group'] = ChannelGroup(**kargs)
            gp['channel_group']['ch_nr'] = ch_cntr
            gp['size'] = cycles_nr * (offset >> 3)

            #data group
            kargs = {'block_len': DG32_BLOCK_SIZE if self.version in ('3.20', '3.30') else DG31_BLOCK_SIZE}
            gp['data_group'] = DataGroup(**kargs)

            #data block

            types = dtype(types)

            gp['types'] = types
            gp['parents'] = parents

            samples = fromarrays(arrays, dtype=types)
            block = samples.tostring()

            if self.load_measured_data:
                gp['data_location'] = LOCATION_MEMORY
                kargs = {'data': block}
                gp['data_block'] = DataBlock(**kargs)
            else:
                gp['data_location'] = LOCATION_TEMPORARY_FILE
                if self._tempfile is None:
                    self._tempfile = TemporaryFile()
                self._tempfile.seek(0, SEEK_END)
                data_address = self._tempfile.tell()
                gp['data_group']['data_block_addr'] = data_address
                self._tempfile.write(block)

            # data group trigger
            gp['trigger'] = [None, None]

    def close(self):
        """ if the MDF was created with load_measured_data=False and new channels
        have been appended, then this must be called just before the object is not
        used anymore to clean-up the temporary file"""
        if self.load_measured_data == False and self._tempfile is not None:
            self._tempfile.close()

    def get(self, name=None, group=None, index=None, raster=None, samples_only=False):
        """Gets channel samples.
        Channel can be specified in two ways:

        * using the first positional argument *name*

            * if there are multiple occurances for this channel then the *group* and *index* arguments can be used to select a specific group.
            * if there are multiple occurances for this channel and either the *group* or *index* arguments is None then a warning is issued

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
            returns *Signal* if *samples_only*=*False* (default option), otherwise returns numpy.array.
            The *Signal* samples are:

                * numpy recarray for channels that have CDBLOCK or BYTEARRAY type channels
                * numpy array for all the rest

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
                if group is None or index is None:
                    gp_nr, ch_nr = self.channels_db[name][0]
                    if len(self.channels_db[name]) > 1:
                        warnings.warn('Multiple occurances for channel "{}". Using first occurance from data group {}. Provide both "group" and "index" arguments to select another data group'.format(name, gp_nr))
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
        dependency_block = grp['channel_dependencies'][ch_nr]
        cycles_nr = grp['channel_group']['cycles_nr']

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

        info = None

        # check if this is a channel array
        if dependency_block:
            if dependency_block['dependency_type'] == DEPENDENCY_TYPE_VECTOR:
                shape = (dependency_block['sd_nr'], )
            elif dependency_block['dependency_type'] >= DEPENDENCY_TYPE_NDIM:
                shape = []
                i = 0
                while True:
                    try:
                        dim = dependency_block['dim_{}'.format(i)]
                        shape.append(dim)
                        i += 1
                    except KeyError:
                        break
                shape = shape[::-1]

            record_shape = tuple(shape)

            referenced_channels = [ch for ch, gp_ in dependency_block.referenced_channels]
            arrays = [self.get(ch.name, samples_only=True) for ch in referenced_channels]
            if cycles_nr:
                shape.insert(0, cycles_nr)

            vals = column_stack(arrays).flatten().reshape(tuple(shape))

            arrays = [vals, ]
            types = [ (channel.name, vals.dtype, record_shape), ]
            types = dtype(types)
            vals = fromarrays(arrays, dtype=types)

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
                             comment=comment,
                             info=info)

                if raster and t:
                    tx = linspace(0, t[-1], int(t[-1] / raster))
                    res = res.interp(tx)
                return res
        else:
            # get channel values
            parent, bit_offset = parents[ch_nr]
            vals = record[parent]

            if bit_offset:
                vals = vals >> bit_offset
            bits = channel['bit_count']
            if bits % 8:
                vals = vals & ((1<<bits) - 1)

            info = None

            conversion_type = CONVERSION_TYPE_NONE if conversion is None else conversion['conversion_type']

            if conversion_type == CONVERSION_TYPE_NONE:

                if channel['data_type'] == DATA_TYPE_STRING:
                    vals = [val.tobytes() for val in vals]
                    vals = array([x.decode('latin-1').strip(' \n\t\x00') for x in vals])
                    if PYVERSION == 2:
                        vals = array([str(val) for val in vals])

                    vals = encode(vals, 'latin-1')

                elif channel['data_type'] == DATA_TYPE_BYTEARRAY:
                    arrays = [vals, ]
                    types = [ (channel.name, vals.dtype, vals.shape[1:]), ]
                    types = dtype(types)
                    vals = fromarrays(arrays, dtype=types)

            elif conversion_type == CONVERSION_TYPE_LINEAR:
                a = conversion['a']
                b = conversion['b']
                if (a, b) != (1, 0):
                    vals = vals * a
                    if b:
                        vals += b

            elif conversion_type in (CONVERSION_TYPE_TABI, CONVERSION_TYPE_TABX):
                nr = conversion['ref_param_nr']
                raw = array([conversion['raw_{}'.format(i)] for i in range(nr)])
                phys = array([conversion['phys_{}'.format(i)] for i in range(nr)])
                if conversion_type == CONVERSION_TYPE_TABI:
                    vals = interp(vals, raw, phys)
                else:
                    idx = searchsorted(raw, vals)
                    idx = clip(idx, 0, len(raw) - 1)
                    vals = phys[idx]

            elif conversion_type == CONVERSION_TYPE_VTAB:
                nr = conversion['ref_param_nr']
                raw = array([conversion['param_val_{}'.format(i)] for i in range(nr)])
                phys = array([conversion['text_{}'.format(i)] for i in range(nr)])
                info = {'raw': raw, 'phys': phys}

            elif conversion_type == CONVERSION_TYPE_VTABR:
                nr = conversion['ref_param_nr']

                texts = array([grp['texts']['conversion_tab'][ch_nr].get('text_{}'.format(i), {}).get('text', b'') for i in range(nr)])
                lower = array([conversion['lower_{}'.format(i)] for i in range(nr)])
                upper = array([conversion['upper_{}'.format(i)] for i in range(nr)])
                info = {'lower': lower, 'upper': upper, 'phys': texts}

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
                    vals = func(((vals - P7) * P6 - P3) / P1) / P2
                elif P1 == 0:
                    vals = func((P3 / (vals - P7) - P6) / P4) / P5
                else:
                    raise ValueError('wrong conversion type {}'.format(conversion_type))

            elif conversion_type == CONVERSION_TYPE_RAT:
                P1 = conversion['P1']
                P2 = conversion['P2']
                P3 = conversion['P3']
                P4 = conversion['P4']
                P5 = conversion['P5']
                P6 = conversion['P6']
                X = vals
                vals = evaluate('(P1 * X**2 + P2 * X + P3) / (P4 * X**2 + P5 * X + P6)')

            elif conversion_type == CONVERSION_TYPE_POLY:
                P1 = conversion['P1']
                P2 = conversion['P2']
                P3 = conversion['P3']
                P4 = conversion['P4']
                P5 = conversion['P5']
                P6 = conversion['P6']
                X = vals
                vals = evaluate('(P2 - (P4 * (X - P5 -P6))) / (P3* (X - P5 - P6) - P1)')

            elif conversion_type == CONVERSION_TYPE_FORMULA:
                formula = conversion['formula'].decode('latin-1').strip(' \n\t\x00')
                X1 = vals
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
                                 info=info)

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
                    comment = trigger_text['text'].decode('latin-1').strip(' \n\t\x00')
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
        info['version'] = self.identification['version_str'].strip(b'\x00').decode('latin-1').strip(' \n\t\x00')
        info['author'] = self.header['author'].strip(b'\x00').decode('latin-1').strip(' \n\t\x00')
        info['organization'] = self.header['organization'].strip(b'\x00').decode('latin-1').strip(' \n\t\x00')
        info['project'] = self.header['project'].strip(b'\x00').decode('latin-1').strip(' \n\t\x00')
        info['subject'] = self.header['subject'].strip(b'\x00').decode('latin-1').strip(' \n\t\x00')
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

    def save(self, dst='', overwrite=False, compression=0):
        """Save MDF to *dst*. If *dst* is not provided the the destination file name is
        the MDF name. If overwrite is *True* then the destination file is overwritten,
        otherwise the file name is appened with '_xx', were 'xx' is the first conter that produces a new
        file name (that does not already exist in the filesystem)

        Parameters
        ----------
        dst : str
            destination file name, Default ''
        overwrite : bool
            overwrite flag, default *False*
        compression : int
            does nothing for mdf version3; introduced here to share the same API as mdf version 4 files

        """

        if self.file_history is None:
            self.file_history = TextBlock(text='''<FHcomment>
<TX>created</TX>
<tool_id>asammdf</tool_id>
<tool_vendor> </tool_vendor>
<tool_version>2.4.1</tool_version>
</FHcomment>''')
        else:
            text = self.file_history['text'] + '\n{}: updated byt Python script'.format(time.asctime()).encode('latin-1')
            self.file_history = TextBlock(text=text)

        if self.name is None and dst == '':
            raise MdfException('Must specify a destination file name for MDF created from scratch')

        dst = dst if dst else self.name
        if overwrite == False:
            if os.path.isfile(dst):
                cntr = 0
                while True:
                    name = os.path.splitext(dst)[0] + '_{}.mdf'.format(cntr)
                    if not os.path.isfile(name):
                        break
                    else:
                        cntr += 1
                warnings.warn('Destination file "{}" already exists and "overwrite" is False. Saving MDF file as "{}"'.format(dst, name))
                dst = name

        # all MDF blocks are appended to the blocks list in the order in which
        # they will be written to disk. While creating this list, all the relevant
        # block links are updated so that once all blocks have been added to the list
        # they can simply be written (using the bytes protocol).
        # DataGroup blocks are written first after the identification and header blocks.
        # When load_measured_data=False we need to restore the original data block addresses
        # within the data group block. This is needed to allow further work with the object
        # after the save method call (eq. new calls to get method). Since the data group blocks
        # are written first, it is safe to restor the original links when the data blocks
        # are written. For lado_measured_data=False, the blocks list will contain a tuple
        # instead of a DataBlock instance; the tuple will have the reference to the
        # data group object and the original link to the data block in the soource MDF file.

        with open(dst, 'wb') as dst:
            #store unique texts and their addresses
            defined_texts = {}
            # list of all blocks
            blocks = []

            address = 0

            blocks.append(self.identification)
            address += ID_BLOCK_SIZE

            blocks.append(self.header)
            address += self.header['block_len']

            self.file_history.address = address
            blocks.append(self.file_history)
            address += self.file_history['block_len']

            # DataGroup
            # put them first in the block list so they will be written first to disk
            # this way, in case of load_measured_data=False, we can safely restore
            # the original data block address
            for gp in self.groups:
                dg = gp['data_group']
                blocks.append(dg)
                dg.address = address
                address += dg['block_len']
            for i, dg in enumerate(self.groups[:-1]):
                dg['data_group']['next_dg_addr'] = self.groups[i+1]['data_group'].address
            self.groups[-1]['data_group']['next_dg_addr'] = 0

            for index, gp in enumerate(self.groups):
                gp_texts = gp['texts']

                # Texts
                for item_list in gp_texts.values():
                    for my_dict in item_list:
                        for key, tx_block in my_dict.items():
                            #text blocks can be shared
                            text = tx_block['text']
                            if text in defined_texts:
                                tx_block.address = defined_texts[text]
                            else:
                                defined_texts[text] = address
                                tx_block.address = address
                                blocks.append(tx_block)
                                address += tx_block['block_len']

                # ChannelConversions
                cc = gp['channel_conversions']
                for i, conv in enumerate(cc):
                    if conv:
                        conv.address = address
                        if conv['conversion_type'] == CONVERSION_TYPE_VTABR:
                            for key, item in gp_texts['conversion_tab'][i].items():
                                conv[key] = item.address

                        blocks.append(conv)
                        address += conv['block_len']

                # Channel Extension
                cs = gp['channel_extensions']
                for source in cs:
                    if source:
                        source.address = address
                        blocks.append(source)
                        address += source['block_len']

                # Channel Dependency
                cd = gp['channel_dependencies']
                for dep in cd:
                    if dep:
                        dep.address = address
                        blocks.append(dep)
                        address += dep['block_len']

                # Channels
                ch_texts = gp_texts['channels']
                for i, channel in enumerate(gp['channels']):
                    channel.address = address
                    channel_texts = ch_texts[i]

                    blocks.append(channel)
                    address += CN_BLOCK_SIZE

                    for key in ('long_name_addr', 'comment_addr', 'display_name_addr'):
                        text_block = channel_texts.get(key, None)
                        channel[key] = 0 if text_block is None else text_block.address

                    channel['conversion_addr'] = cc[i].address if cc[i] else 0
                    channel['source_depend_addr'] = cs[i].address if cs[i] else 0
                    if cd[i]:
                        channel['ch_depend_addr'] = cd[i].address
                    else:
                        channel['ch_depend_addr'] = 0

                for channel, next_channel in pair(gp['channels']):
                    channel['next_ch_addr'] = next_channel.address
                next_channel['next_ch_addr'] = 0

                # ChannelGroup
                cg = gp['channel_group']
                cg.address = address
                blocks.append(cg)
                address += cg['block_len']

                cg['first_ch_addr'] = gp['channels'][0].address
                cg['next_cg_addr'] = 0
                if 'comment_addr' in gp['texts']['channel_group'][0]:
                    cg['comment_addr'] = gp_texts['channel_group'][0]['comment_addr'].address

                # TriggerBLock
                trigger, trigger_text = gp['trigger']
                if trigger:
                    if trigger_text:
                        trigger_text.address = address
                        blocks.append(trigger_text)
                        address += trigger_text['block_len']
                        trigger['comment_addr'] = trigger_text.address
                    else:
                        trigger['comment_addr'] = 0

                    trigger.address = address
                    blocks.append(trigger)
                    address += trigger['block_len']

                # DataBlock
                original_data_addr = gp['data_group']['data_block_addr']
                gp['data_group']['data_block_addr'] = address if gp['size'] else 0
                address += gp['size']
                if self.load_measured_data:
                    blocks.append(gp['data_block'])
                else:
                    # trying to call bytes([gp, address]) will result in an exception
                    # that be used as a flag for non existing data block in case
                    # of load_measured_data=False, the address is the actual address
                    # of the data group's data within the original file
                    blocks.append([gp, original_data_addr])

            # update referenced channels addresses within the channel dependecies
            for gp in self.groups:
                for dep in gp['channel_dependencies']:
                    if dep:
                        for i, (ch, grp) in enumerate(dep.referenced_channels):
                            dep['ch_{}'.format(i)] = ch.address
                            dep['cg_{}'.format(i)] = grp['channel_group'].address
                            dep['dg_{}'.format(i)] = grp['data_group'].address

            # DataGroup
            for gp in self.groups:
                gp['data_group']['first_cg_addr'] = gp['channel_group'].address
                gp['data_group']['trigger_addr'] = gp['trigger'][0].address if gp['trigger'][0] else 0

            if self.groups:
                self.header['first_dg_addr'] = self.groups[0]['data_group'].address
                self.header['dg_nr'] = len(self.groups)
                self.header['comment_addr'] = self.file_history.address
                self.header['program_addr'] = 0

            write = dst.write
            for block in blocks:
                try:
                    write(bytes(block))
                except:
                    # this will only be executed for data blocks when load_measured_data=False
                    gp, address = block
                    # restore data block address from original file so that
                    # future calls to get will still work after the save
                    gp['data_group']['data_block_addr'] = address
                    data = self._load_group_data(gp)
                    write(data)


if __name__ == '__main__':
    pass

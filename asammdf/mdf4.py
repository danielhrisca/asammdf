"""
ASAM MDF version 4 file format module
"""
from __future__ import print_function, division
import sys
PYVERSION = sys.version_info[0]

import time
import warnings
import os
from struct import unpack, unpack_from
from functools import reduce
from collections import defaultdict
from hashlib import md5

from numpy import (interp, linspace, dtype, amin, amax, array_equal,
                   array, searchsorted, clip, union1d, float64, frombuffer,
                   uint8,
                   issubdtype, flexible)
from numexpr import evaluate
from numpy.core.records import fromstring, fromarrays

from .v4blocks import (AttachmentBlock,
                       Channel,
                       ChannelGroup,
                       ChannelConversion,
                       DataBlock,
                       DataZippedBlock,
                       DataGroup,
                       DataList,
                       FileHistory,
                       FileIdentificationBlock,
                       HeaderBlock,
                       HeaderList,
                       SignalDataBlock,
                       SourceInformation,
                       TextBlock)

from .v4constants import *
from .utils import MdfException, get_fmt, fmt_to_datatype, pair
from .signal import Signal

if PYVERSION == 2:
    def bytes(obj):
        return obj.__bytes__()

__all__ = ['MDF4', ]


class MDF4(object):
    """If the *name* exist it will be loaded otherwise an empty file will be created that can be later saved to disk

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
        mdf file version ('4.00', '4.10', '4.11'); default '4.00'

    Attributes
    ----------
    name : string
        mdf file name
    groups : list
        list of data groups
    header : HeaderBlock
        mdf file header
    file_history : list
        list of (FileHistory, TextBlock) pairs
    comment : TextBlock
        mdf file comment
    identification : FileIdentificationBlock
        mdf file start block
    load_measured_data : bool
        load measured data option
    compression : bool
        measured data compression option
    version : int
        mdf version
    channels_db : dict
        used for fast channel access by name; for each name key the value is a (group index, channel index) tuple
    masters_db : dict
        used for fast master channel access; for each group index key the value is the master channel index

    """
    def __init__(self, name=None, load_measured_data=True, compression=False, version='4.00'):
        self.groups = []
        self.header = None
        self.identification = None
        self.file_history = []
        self.file_comment = None
        self.name = name
        self.load_measured_data = load_measured_data
        self.channels_db = {}
        self.masters_db = {}
        self.compression = compression
        self.attachments = []

        if name and os.path.isfile(name):
            with open(self.name, 'rb') as file_stream:
                self._read(file_stream)
        else:
            self.load_measured_data = True

            self.header = HeaderBlock()
            self.identification = FileIdentificationBlock(version=version)
            self.version = version

    def _read(self, file_stream):
        dg_cntr = 0

        self.identification = FileIdentificationBlock(file_stream=file_stream)
        self.version = self.identification['version_str'].decode('utf-8').strip(' ').strip('\x00')
        self.header = HeaderBlock(address=0x40, file_stream=file_stream)

        # read file comment
        if self.header['comment_addr']:
            self.file_comment = TextBlock(address=self.header['comment_addr'], file_stream=file_stream)

        # read file history
        fh_addr = self.header['file_history_addr']
        while fh_addr:
            fh = FileHistory(address=fh_addr, file_stream=file_stream)
            try:
                fh_text = TextBlock(address=fh['comment_addr'], file_stream=file_stream)
            except:
                print(self.name)
                raise
            self.file_history.append((fh, fh_text))
            fh_addr = fh['next_fh_addr']

        # read attachments
        at_addr = self.header['first_attachment_addr']
        while at_addr:
            texts = {}
            at_block = AttachmentBlock(address=at_addr, file_stream=file_stream)
            for key in ('file_name_addr', 'mime_addr', 'comment_addr'):
                addr = at_block[key]
                if addr:
                    texts[key] = TextBlock(address=addr, file_stream=file_stream)

            self.attachments.append((at_block, texts))
            at_addr = at_block['next_at_addr']


        # go to first date group and read each data group sequentially
        dg_addr = self.header['first_dg_addr']

        while dg_addr:
            new_groups = []
            group = DataGroup(address=dg_addr, file_stream=file_stream)

            # go to first channel group of the current data group
            cg_addr = group['first_cg_addr']

            cg_nr = 0

            while cg_addr:
                cg_nr += 1

                grp = {}
                new_groups.append(grp)

                grp['channels'] = []
                grp['channel_conversions'] = []
                grp['channel_sources'] = []
                grp['signal_data'] = []
                # channel_group is lsit to allow uniform handling of all texts in save method
                grp['texts'] = {'channels': [], 'sources': [], 'conversions': [], 'conversion_tab': [], 'channel_group': []}

                # read each channel group sequentially
                channel_group = grp['channel_group'] = ChannelGroup(address=cg_addr, file_stream=file_stream)
                # read acquisition name and comment for current channel group
                channel_group_texts = {}
                grp['texts']['channel_group'].append(channel_group_texts)

                grp['data_group'] = DataGroup(address=dg_addr, file_stream=file_stream)

                for key in ('acq_name_addr', 'comment_addr'):
                    address = channel_group[key]
                    if address:
                        channel_group_texts[key] = TextBlock(address=address, file_stream=file_stream)

                # go to first channel of the current channel group
                ch_addr = channel_group['first_ch_addr']
                ch_cntr = 0

                # Read channels by walking recursively in the channel group
                # starting from the first channel
                self._read_channels(ch_addr, grp, file_stream, dg_cntr, ch_cntr)

                cg_addr = channel_group['next_cg_addr']

                if cg_addr and self.load_measured_data == False:
                    raise MdfException('Reading unsorted file with load_measured_data option set to False is not supported')

            if self.load_measured_data:
                size = 0
                record_id_nr = group['record_id_len'] if group['record_id_len'] <= 2 else 0

                cg_size = {}
                cg_data = defaultdict(list)
                for grp in new_groups:
                    if grp['channel_group']['flags'] == 0:
                        cg_size[grp['channel_group']['record_id']] = grp['channel_group']['samples_byte_nr']
                    else:
                        # VLDS flags
                        cg_size[grp['channel_group']['record_id']] = 0

                # go to the first data block of the current data group
                dat_addr = group['data_block_addr']
                data = self._read_data_block(address=dat_addr, file_stream=file_stream)

                if cg_nr == 1:
                    kargs = {'data': data, 'compression': self.compression}
                    new_groups[0]['data_block'] = DataBlock(**kargs)
                else:
                    i = 0
                    size = len(data)
                    while i < size:
                        rec_id = data[i]
                        # skip redord id
                        i += 1
                        rec_size = cg_size[rec_id]
                        if rec_size:
                            rec_data = data[i: i+rec_size]
                            cg_data[rec_id].append(rec_data)
                        else:
                            # as shown bby mdfvalidator rec size is first byte after rec id + 3
                            rec_size = unpack('<I', data[i: i+3])[0]
                            i += 4
                            rec_data = data[i: i + rec_size]
                            cg_data[rec_id].append(rec_data)
                        # if 2 record id's are used skip also the second one
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

            dg_addr = group['next_dg_addr']
            dg_cntr += 1

    def _read_channels(self, ch_addr, grp, file_stream, dg_cntr, ch_cntr):
        channels = grp['channels']
        while ch_addr:
            # read channel block and create channel object
            channel = Channel(address=ch_addr, file_stream=file_stream)
            if channel['component_addr'] != 0:
                ch_cntr = self._read_channels(channel['component_addr'], grp, file_stream, dg_cntr, ch_cntr)
            else:
                channels.append(channel)

                # append channel signal data if load_measured_data allows it
                if self.load_measured_data:
                    ch_data_addr = channel['data_block_addr']
                    signal_data = self._read_agregated_signal_data(address=ch_data_addr, file_stream=file_stream)
                    if signal_data:
                        grp['signal_data'].append(SignalDataBlock(data=signal_data))
                    else:
                        grp['signal_data'].append(None)
                else:
                    grp['signal_data'].append(None)

                # read conversion block and create channel conversion object
                address = channel['conversion_addr']
                if address:
                    conv = ChannelConversion(address=address, file_stream=file_stream)
                else:
                    conv = None
                grp['channel_conversions'].append(conv)

                conv_tabx_texts = {}
                grp['texts']['conversion_tab'].append(conv_tabx_texts)
                if conv and conv['conversion_type'] in (CONVERSION_TYPE_TABX, CONVERSION_TYPE_RTABX, CONVERSION_TYPE_TTAB):
                    # link_nr - common links (4) - default text link (1)
                    for i in range(conv['links_nr'] - 4 - 1):
                        address = conv['text_{}'.format(i)]
                        if address:
                            conv_tabx_texts['text_{}'.format(i)] = TextBlock(address=address, file_stream=file_stream)
                    address = conv.get('default_addr', 0)
                    if address:
                        file_stream.seek(address, SEEK_START)
                        blk_id = file_stream.read(4)
                        if blk_id == b'##TX':
                            conv_tabx_texts['default_addr'] = TextBlock(address=address, file_stream=file_stream)
                        elif blk_id == b'##CC':
                            conv_tabx_texts['default_addr'] = ChannelConversion(address=address, file_stream=file_stream)
                            conv_tabx_texts['default_addr'].text_str = str(time.clock())

                            conv['unit_addr'] = conv_tabx_texts['default_addr']['unit_addr']
                            conv_tabx_texts['default_addr']['unit_addr'] = 0
                elif conv and conv['conversion_type'] == CONVERSION_TYPE_TRANS:
                    # link_nr - common links (4) - default text link (1)
                    for i in range((conv['links_nr'] - 4 - 1 ) //2):
                        for key in ('input_{}_addr'.format(i), 'output_{}_addr'.format(i)):
                            address = conv[key]
                            if address:
                                conv_tabx_texts[key] = TextBlock(address=address, file_stream=file_stream)
                    address = conv['default_addr']
                    if address:
                        conv_tabx_texts['default_addr'] = TextBlock(address=address, file_stream=file_stream)

                if self.load_measured_data:
                    # read source block and create source information object
                    source_texts = {}
                    address = channel['source_addr']
                    if address:
                        source = SourceInformation(address=address, file_stream=file_stream)
                        grp['channel_sources'].append(source)
                        grp['texts']['sources'].append(source_texts)
                        # read text fields for channel sources
                        for key in ('name_addr', 'path_addr', 'comment_addr'):
                            address = source[key]
                            if address:
                                source_texts[key] = TextBlock(address=address, file_stream=file_stream)
                    else:
                        grp['channel_sources'].append(None)
                        grp['texts']['sources'].append(source_texts)
                else:
                    grp['channel_sources'].append(None)
                    grp['texts']['sources'].append({})

                # read text fields for channel conversions
                conv_texts = {}
                grp['texts']['conversions'].append(conv_texts)
                for key in ('name_addr', 'unit_addr', 'comment_addr', 'formula_addr'):
                    if conv is not None:
                        address = conv.get(key, 0)
                        if address:
                            conv_texts[key] = TextBlock(address=address, file_stream=file_stream)

                # read text fields for channel
                channel_texts = {}
                grp['texts']['channels'].append(channel_texts)
                for key in ('name_addr', 'comment_addr', 'unit_addr'):
                    address = channel[key]
                    if address:
                        channel_texts[key] = TextBlock(address=address, file_stream=file_stream)

                # update channel object name and block_size attributes
                channel.name = channel_texts['name_addr'].text_str
                self.channels_db[channel.name] = (dg_cntr, ch_cntr)

                if channel['channel_type'] in (CHANNEL_TYPE_MASTER, CHANNEL_TYPE_VIRTUAL_MASTER):
                    self.masters_db[dg_cntr] = ch_cntr

                ch_cntr += 1

            # go to next channel of the current channel group
            ch_addr = channel['next_ch_addr']

        return ch_cntr

    def _read_data_block(self, address, file_stream):
        """read and agregate data blocks for a given data group

        Returns
        -------
        data : bytes
            agregated raw data
        """
        if address:
            file_stream.seek(address, SEEK_START)
            id_string = file_stream.read(4)
            if id_string == b'##DT':
                data = DataBlock(address=address, file_stream=file_stream)['data']
            elif id_string == b'##DZ':
                data = DataZippedBlock(address=address, file_stream=file_stream)['data']
            elif id_string == b'##DL':
                data = []
                while address:
                    dl = DataList(address=address, file_stream=file_stream)
                    for i in range(dl['links_nr'] - 1):
                        addr = dl['data_block_addr{}'.format(i)]
                        file_stream.seek(addr, SEEK_START)
                        id_string = file_stream.read(4)
                        if id_string == b'##DT':
                            data.append(DataBlock(file_stream=file_stream, address=addr)['data'])
                        elif id_string == b'##DZ':
                            data.append(DataZippedBlock(address=addr, file_stream=file_stream)['data'])
                        elif id_string == b'##DL':
                            data.append(self._read_data_block(address=addr, file_stream=file_stream))
                    address = dl['next_dl_addr']
                data = b''.join(data)
            elif id_string == b'##HL':
                hl = HeaderList(address=address, file_stream=file_stream)
                return self._read_data_block(address=hl['first_dl_addr'], file_stream=file_stream)
        else:
            data = b''
        return data

    def _read_agregated_signal_data(self, address, file_stream):
        if address:

            file_stream.seek(address, SEEK_START)
            blk_id = file_stream.read(4)
            if blk_id == b'##SD':
                data = SignalDataBlock(address=address, file_stream=file_stream)['data']
            elif blk_id == b'##DZ':
                data = DataZippedBlock(address=address, file_stream=file_stream)['data']
            elif blk_id == b'##DL':
                data = []
                while address:
                    # the data list will contain only links to SDBLOCK's
                    data_list = DataList(address=address, file_stream=file_stream)
                    nr = data_list['links_nr']
                    # aggregate data from all SDBLOCK
                    for i in range(nr-1):
                        addr = data_list['data_block_addr{}'.format(i)]
                        file_stream.seek(addr, SEEK_START)
                        blk_id = file_stream.read(4)
                        if blk_id == b'##SD':
                            data.append(SignalDataBlock(address=addr, file_stream=file_stream)['data'])
                        elif blk_id == b'##DZ':
                            data.append(DataZippedBlock(address=addr, file_stream=file_stream)['data'])
                        else:
                            warnings.warn('Expected SD, DZ or DL block at {} but found id="{}"'.format(hex(address), blk_id))
                            return
                    address = data_list['next_dl_addr']
                data = b''.join(data)
            else:
                warnings.warn('Expected SD, DL or DZ block at {} but found id="{}"'.format(hex(address), blk_id))
                return
        else:
            data = b''

        return data

    def append(self, signals, source_info='Python'):
        """Appends a new data group.

        Parameters
        ----------
        signals : list
            list on *Signal* objects
        acquisition_info : str
            acquisition information; default 'Python'

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
        >>> mdf = MDF4('new.mf4')
        >>> mdf.append([s1, s2, s3], 'created by asammdf v1.1.0')
        >>> # case 2: VTAB conversions from channels inside another file
        >>> mdf1 = MDF4('in.mf4')
        >>> ch1 = mdf1.get("Channel1_VTAB")
        >>> ch2 = mdf1.get("Channel2_VTABR")
        >>> sigs = [ch1, ch2]
        >>> mdf2 = MDF4('out.mf4')
        >>> mdf2.append(sigs, 'created by asammdf v1.1.0')

        """
        if self.load_measured_data == False:
            warnings.warn("Can't append if load_measurement_data option is False")
            return

        if not signals:
            warnings.warn("Must provide at least one Signal object in the input list for append")
            return

        signals_nr = len(signals)
        dg_cntr = len(self.groups)
        self.groups.append({})
        gp = self.groups[-1]

        # check if all signals have the same time base
        t_ = signals[0].timestamps
        for s in signals[1:]:
            if not array_equal(s.timestamps, t_):
                different = True
                break
        else:
            different = False

        # computed union of all time bases
        if different:
            times = [s.timestamps for s in signals]
            t = reduce(union1d, times).flatten().astype(float64)
            signals = [s.interp(t) for s in signals]
            times = None
        else:
            t = t_

        cycles_nr = len(t)

        t_type, t_size = fmt_to_datatype(t.dtype, version=4)

        gp['channels'] = gp_channels = []
        gp['channel_conversions'] = gp_conv = []
        gp['channel_sources'] = gp_source = []
        # SDBLOCKS are not used when appending
        # so we get None for each signal, +1 for the master channel
        gp['signal_data'] = [None,] * (signals_nr + 1)
        gp['texts'] = gp_texts = {'channels': [], 'sources': [], 'conversions': [], 'conversion_tab': [], 'channel_group': []}

        # time channel texts
        for key, item in gp['texts'].items():
            item.append({})
        gp_texts['channels'][-1]['name_addr'] = TextBlock.from_text('t')
        gp_texts['conversions'][-1]['unit_addr'] = TextBlock.from_text('s')

        gp_texts['sources'][-1]['name_addr'] = TextBlock.from_text(source_info)
        gp_texts['sources'][-1]['path_addr'] = TextBlock.from_text(source_info)
        gp_texts['channel_group'][-1]['acq_name_addr'] = TextBlock.from_text(source_info)
        gp_texts['channel_group'][-1]['comment_addr'] = TextBlock.from_text(source_info)

        # channels texts
        for s in signals:
            for key, item in gp['texts'].items():
                item.append({})
            gp_texts['channels'][-1]['name_addr'] = TextBlock.from_text(s.name)
            if s.unit:
                gp_texts['conversions'][-1]['unit_addr'] = TextBlock.from_text(s.unit)
            gp_texts['sources'][-1]['name_addr'] = TextBlock.from_text(source_info)
            gp_texts['sources'][-1]['path_addr'] = TextBlock.from_text(source_info)

        # conversion for time channel
        kargs = {'conversion_type': CONVERSION_TYPE_NON,
                 'min_phy_value': t[0] if cycles_nr else 0,
                 'max_phy_value': t[-1] if cycles_nr else 0}
        gp_conv.append(ChannelConversion(**kargs))
        gp_texts['conversion_tab'].append({})

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

        # create channel conversions from the Signal's conversion attribute
        for idx, s in enumerate(signals):
            conv = s.conversion
            conv_texts_tab = gp_texts['conversion_tab'][idx+1]
            if conv:
                conv_type = conv['type']
                if conv_type == CONVERSION_TYPE_TABX:
                    kargs = {}
                    kargs['conversion_type'] = CONVERSION_TYPE_TABX
                    raw = conv['raw']
                    phys = conv['phys']
                    for i, (r_, p_) in enumerate(zip(raw, phys)):
                        kargs['text_{}'.format(i)] = 0
                        kargs['val_{}'.format(i)] = r_
                        conv_texts_tab['text_{}'.format(i)] = TextBlock.from_text(p_)
                    if conv.get('default', b''):
                        conv_texts_tab['default_addr'] = TextBlock.from_text(conv['default'])
                    kargs['default_addr'] = 0
                    kargs['links_nr'] = len(raw) + 5
                elif conv_type == CONVERSION_TYPE_RTABX:
                    kargs = {}
                    kargs['conversion_type'] = CONVERSION_TYPE_RTABX
                    lower = conv['lower']
                    upper = conv['upper']
                    texts = conv['phys']
                    kargs['ref_param_nr'] = len(upper)
                    kargs['default_addr'] = conv.get('default', 0)
                    kargs['links_nr'] = len(lower) + 5

                    for i, (u_, l_, t_) in enumerate(zip(upper, lower, texts)):
                        kargs['lower_{}'.format(i)] = l_
                        kargs['upper_{}'.format(i)] = u_
                        kargs['text_{}'.format(i)] = 0
                        conv_texts_tab['text_{}'.format(i)] = TextBlock.from_text(t_)
                    if conv.get('default', b''):
                        conv_texts_tab['default_addr'] = TextBlock.from_text(conv['default'])
                    kargs['default_addr'] = 0

                else:
                     kargs = {'conversion_type': CONVERSION_TYPE_NON,
                              'min_phy_value': min_max[idx][0],
                              'max_phy_value': min_max[idx][1]}
                gp_conv.append(ChannelConversion(**kargs))
            else:
                sigmin, sigmax = min_max[idx]
                kargs = {'conversion_type': CONVERSION_TYPE_NON,
                         'min_phy_value': sigmin if sigmin<=sigmax else 0,
                         'max_phy_value': sigmax if sigmin<=sigmax else 0,}
                gp_conv.append(ChannelConversion(**kargs))


        #source for channels
        for i in range(signals_nr + 1):
            gp_source.append(SourceInformation())

        #time channel
        kargs = {'channel_type': CHANNEL_TYPE_MASTER,
                 'data_type': t_type,
                 'sync_type': 1,
                 'byte_offset': 0,
                 'bit_count': t_size,
                 'min_raw_value': t[0] if cycles_nr else 0,
                 'max_raw_value' : t[-1] if cycles_nr else 0,
                 'lower_limit' : t[0] if cycles_nr else 0,
                 'upper_limit' : t[-1] if cycles_nr else 0}
        ch = Channel(**kargs)
        ch.name = 't'
        gp_channels.append(ch)
        self.masters_db[dg_cntr] = 0

        #channels
        sig_dtypes = [sig.samples.dtype for sig in signals]
        sig_formats = [fmt_to_datatype(typ, version=4) for typ in sig_dtypes]
        offset = t_size // 8
        ch_cntr = 1
        for (sigmin, sigmax), (sig_type, sig_size), name in zip(min_max, sig_formats, [sig.name for sig in signals]):
            byte_size = max(sig_size // 8, 1)
            kargs = {'channel_type': CHANNEL_TYPE_VALUE,
                     'bit_count': sig_size,
                     'byte_offset': offset,
                     'bit_offset' : 0,
                     'data_type': sig_type,
                     'min_raw_value': sigmin if sigmin<=sigmax else 0,
                     'max_raw_value' : sigmax if sigmin<=sigmax else 0,
                     'lower_limit' : sigmin if sigmin<=sigmax else 0,
                     'upper_limit' : sigmax if sigmin<=sigmax else 255}
            ch = Channel(**kargs)
            ch.name = name
            gp_channels.append(ch)
            offset += byte_size
            self.channels_db[name] = (dg_cntr, ch_cntr)
            ch_cntr += 1

        #channel group
        kargs = {'cycles_nr': len(t),
                 'samples_byte_nr': offset}
        gp['channel_group'] = ChannelGroup(**kargs)

        #data block
        types = [('t', t.dtype),]
        types.extend([('sig{}'.format(i), typ) for i, typ in enumerate(sig_dtypes)])

        arrays = [t, ]
        arrays.extend([sig.samples for sig in signals])

        arrays = fromarrays(arrays, dtype=types)
        block = arrays.tostring()

        kargs = {'data': block,
                 'block_len': 24 + len(block),
                 'compression' : self.compression}
        gp['data_block'] = DataBlock(**kargs)

        #data group
        gp['data_group'] = DataGroup()

    def attach(self, data, file_name=None, comment=None, compression=True, mime=r'application/octet-stream'):
        """ attach embedded attachment as application/octet-stream

        Parameters
        ----------
        data : bytes
            data to be attached
        file_name : str
            string file name
        comment : str
            attachment comment
        compression : bool
            use compression for embedded attachment data
        mime : str
            mime type string

        """
        creator_index = len(self.file_history)
        fh = FileHistory()
        fh_text = TextBlock.from_text("""<FHcomment>
	<TX>Added new embedded attachment from {}</TX>
	<tool_id>asammdf</tool_id>
	<tool_vendor>asammdf</tool_vendor>
	<tool_version>2.0.0</tool_version>
</FHcomment>""".format(file_name if file_name else 'bin.bin'), meta=True)

        self.file_history.append((fh, fh_text))

        texts = {}
        texts['mime_addr'] = TextBlock.from_text(mime)
        if comment:
            texts['comment_addr'] = TextBlock.from_text(comment)
        texts['file_name_addr'] = TextBlock.from_text(file_name if file_name else 'bin.bin')
        at_block = AttachmentBlock(data=data, compression=compression)
        at_block['creator_index'] = creator_index
        self.attachments.append((at_block, texts))

    def extract_attachment(self, index):
        """ extract attachemnt *index* data. If it is an embedded attachment, then this method creates the new file according to the attachemnt file name information

        Parameters
        ----------
        index : int
            attachment index

        Returns
        -------
        data : bytes | str
            attachment data

        """
        try:
            current_path = os.getcwd()
            os.chdir(os.path.dirname(self.name))

            attachment, texts = self.attachments[index]
            flags = attachment['flags']

            # for embedded attachments extrat data and create new files
            if flags & FLAG_AT_EMBEDDED:
                data = attachment.extract()

                out_path = os.path.dirname(texts['file_name_addr'].text_str)
                if out_path:
                    if not os.path.exists(out_path):
                        os.makedirs(out_path)

                with open(texts['file_name_addr'].text_str, 'wb') as f:
                    f.write(data)

                return data
            else:
                # for external attachemnts read the files and return the content
                if flags & FLAG_AT_MD5_VALID:
                    data = open(texts['file_name_addr'].text_str, 'rb').read()
                    md5_worker = md5()
                    md5_worker.update(data)
                    md5_sum = md5_worker.digest()
                    if attachment['md5_sum'] == md5_sum:
                        if texts['mime_addr'].text_str.startswith('text'):
                            with open(texts['file_name_addr'].text_str, 'r') as f:
                                data = f.read()
                        return data
                    else:
                        warnings.warn('ATBLOCK md5sum="{}" and external attachment data ({}) md5sum="{}"'.format(self['md5_sum'], texts['file_name_addr'].text_str, md5_sum))
                else:
                    if texts['mime_addr'].text_str.startswith('text'):
                        mode = 'r'
                    else:
                        mode = 'rb'
                    with open(texts['file_name_addr'].text_str, mode) as f:
                        data = f.read()
                    return data
        except Exception as err:
            os.chdir(current_path)
            warnings.warn('Exception during attachment extraction: ' + repr(err))

    def get_master_data(self, name=None, group=None, data=None):
        """get master channel values only. The group is identified by a channel name (*name* argument) or by the index (*group* argument).
        *data* argument is used internally by the *get* method to avoid double work.

        Parameters
        ----------
        name : str
            channel name in target group
        group : int
            group index
        data : bytes
            data groups's raw channel data

        Returns
        -------
        t : numpy.array
            master channel values
        """

        if name is None:
            if group is None:
                raise MdfException('Invalid arguments for "get_master_data" method: must give "name" or "group"')
            else:
                gp_nr = group
                if gp_nr > len(self.groups) - 1 or gp_nr < 0:
                    raise MdfException('Group index out of range')
        else:
            if not name in self.channels_db:
                raise MdfException('Channel "{}" not found'.format(name))
            else:
                gp_nr, _= self.channels_db[name]


        gp = self.groups[gp_nr]

        time_idx = self.masters_db[gp_nr]
        time_ch = gp['channels'][time_idx]
        time_conv = gp['channel_conversions'][time_idx]

        time_size = time_ch['bit_count'] // 8
        t_fmt = get_fmt(time_ch['data_type'], time_size, version=4)
        t_byte_offset, bit_offset = time_ch['byte_offset'], time_ch['bit_offset']
        bits = time_ch['bit_count']
        if bits % 8:
            t_size = bits // 8 + 1
        else:
            t_size = bits // 8

        block_size = gp['channel_group']['samples_byte_nr']

        # get the raw data if it's not provided
        if data is None:
            if not self.load_measured_data:
                with open(self.name, 'rb') as file_stream:
                    # go to the first data block of the current data group
                    dat_addr = gp['data_group']['data_block_addr']
                    data = self._read_data_block(address=dat_addr, file_stream=file_stream)
            else:
                if gp['data_block']:
                    data = gp['data_block']['data']
                else:
                    data = b''

        if time_ch['channel_type'] == CHANNEL_TYPE_MASTER:
            types = dtype( [('', 'a{}'.format(t_byte_offset)),
                            ('t', t_fmt),
                            ('', 'a{}'.format(block_size - t_byte_offset - t_size))] )

            values = fromstring(data, types)

            time_conv_type = CONVERSION_TYPE_NON if time_conv is None else time_conv['conversion_type']
            if time_conv_type == CONVERSION_TYPE_LIN:
                time_a = time_conv['a']
                time_b = time_conv['b']
                t = values['t'] * time_a
                if time_b:
                    t += time_b
            elif time_conv_type == CONVERSION_TYPE_NON:
                t = values['t']

        elif time_ch['channel_type'] == CHANNEL_TYPE_VIRTUAL_MASTER:
            time_a = time_conv['a']
            time_b = time_conv['b']
            cycles = len(data) // block_size
            t = array([t * time_a + time_b for t in range(cycles)], dtype=float64)

        return t

    def get_channel_data(self, name=None, group=None, index=None, data=None, signal_data=None, return_info=False):
        """get channel values. The channel is identified by name (*name* argument) or by the group and channel indexes (*group* and *index* arguments).
        *data* argument is used internally by the *get* method to avoid double work.
        By defaulkt only the channel values are returned. If the *return_info* argument is set then name, unit and conversion info is returned as well

        Parameters
        ----------
        name : str
            channel name in target group
        group : int
            group index
        index : int
            channel index
        data : bytes
            data groups's raw channels data
        signal_data : bytes
            data from SDBLOCKs of VLDS channels
        return_info : bool
            enables returning extra information (name, unit, conversion)

        Returns
        -------
        vals : numpy.array
            channel values; if *return_info* is False
        vals, name, conversion, unit : numpy.array, str, dict, str
            channel values, channel name, channel conversion, channel unit: if *return_info* is True

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
                gp_nr, ch_nr = self.channels_db[name]

        gp = self.groups[gp_nr]
        channel = gp['channels'][ch_nr]

        conversion = gp['channel_conversions'][ch_nr]

        signal_data = gp['signal_data'][ch_nr]
        if signal_data:
            signal_data = signal_data['data']
        else:
            signal_data = b''

        # search for unit in conversion texts
        unit = gp['texts']['conversions'][ch_nr].get('unit_addr', None)
        if unit:
            unit = unit.text_str
        else:
            # search for physical unit in channel texts
            unit = gp['texts']['channels'][ch_nr].get('unit_addr', None)
            if unit:
                unit = unit.text_str
            else:
                unit = ''

        group = gp

        byte_offset, bit_offset = channel['byte_offset'], channel['bit_offset']

        bits = channel['bit_count']
        size = bits + bit_offset

        if size % 8:
            size = size // 8 + 1
        else:
            size = size // 8
        block_size = gp['channel_group']['samples_byte_nr']

#        print(channel, gp_nr, ch_nr, size)

        if data is None:
            if not self.load_measured_data:
                with open(self.name, 'rb') as file_stream:
                    # go to the first data block of the current data group
                    dat_addr = gp['data_group']['data_block_addr']
                    data = self._read_data_block(address=dat_addr, file_stream=file_stream)
            else:
                if gp['data_block']:
                    data = gp['data_block']['data']
                else:
                    data = b''

        if signal_data is None:
            if not self.load_measured_data:
                ch_data_addr = channel['data_block_addr']
                signal_data = self._read_agregated_signal_data(address=ch_data_addr, file_stream=file_stream)
            else:
                signal_data = gp['signal_data'][ch_nr]['data'] if gp['signal_data'][ch_nr] else b''

        ch_fmt = get_fmt(channel['data_type'], size, version=4)

        # for VLSD channel with signal data block change the dtype from string to void
        if signal_data:
            ch_fmt = ch_fmt.replace('S', 'V')

        types = dtype( [('', 'a{}'.format(byte_offset)),
                        ('vals', ch_fmt),
                        ('', 'a{}'.format(block_size - byte_offset - size))] )
#            print(channel.name, types, data)
        values = fromstring(data, types)

        # get channel values
        conversion_type = CONVERSION_TYPE_NON if conversion is None else conversion['conversion_type']
        vals = values['vals']
        if bit_offset:
            vals = vals >> bit_offset
        if bits % 8:
            vals = vals & (2**bits - 1)

        if conversion_type == CONVERSION_TYPE_NON:
            # check if it is VLDS channel type with SDBLOCK
            if signal_data:
                values = []
                vals = vals.tostring()
                if size == 4:
                    fmt = '<u4'
                else:
                    fmt = '<u8'
                vals = frombuffer(vals, dtype=fmt)

                for offset in vals:
                    offset = int(offset)
                    size = unpack_from('<I', signal_data, offset)[0]
                    values.append(signal_data[offset+4: offset+4+size])
                vals = array(values)

            # CANopen date
            elif channel['data_type'] == DATA_TYPE_CANOPEN_DATE:
                vals = vals.tostring()

                types = dtype( [('ms', '<u2'),
                                ('min', '<u1'),
                                ('hour', '<u1'),
                                ('day', '<u1'),
                                ('month', '<u1'),
                                ('year', '<u1')] )
                dates = fromstring(vals, types)

                arrays = []
                arrays.append(dates['ms'])
                # bit 6 and 7 of minutes are reserved
                arrays.append(dates['min'] & 0x3F)
                # only firt 4 bits of hour are used
                arrays.append(dates['hour'] & 0xF)
                # the first 4 bits are the day number
                arrays.append(dates['day'] & 0xF)
                # bit 6 and 7 of month are reserved
                arrays.append(dates['month'] & 0x3F)
                # bit 7 of year is reserved
                arrays.append(dates['year'] & 0x7F)
                # add summer or standard time information for hour
                arrays.append((dates['hour'] & 0x80) >> 7)
                # add day of week information
                arrays.append((dates['day'] & 0xF0) >> 4)

                names = ['ms', 'min', 'hour', 'day', 'month', 'year', 'summer_time', 'day_of_week']
                vals = fromarrays(arrays, names=names)

            # CANopen time
            elif channel['data_type'] == DATA_TYPE_CANOPEN_TIME:
                vals = vals.tostring()

                types = dtype( [('ms', '<u4'),
                                ('days', '<u2')] )
                dates = fromstring(vals, types)

                arrays = []
                # bits 28 to 31 are reserverd for ms
                arrays.append(dates['ms'] & 0xFFFFFFF)
                arrays.append(dates['days'] & 0x3F)

                names = ['ms', 'days']
                vals = fromarrays(arrays, names=names)

            # byte array
            elif channel['data_type'] == DATA_TYPE_BYTEARRAY:
                vals = vals.tostring()
                cols = size
                lines = len(vals) // cols

                vals = frombuffer(vals, dtype=uint8).reshape((lines, cols))

        elif conversion_type == CONVERSION_TYPE_LIN:
            a = conversion['a']
            b = conversion['b']
            if (a, b) == (1, 0):
                if not vals.dtype == ch_fmt:
                    vals = vals.astype(ch_fmt)
            else:
                vals = vals * a
                if b:
                    vals = vals + b

        elif conversion_type == CONVERSION_TYPE_RAT:
            P1 = conversion['P1']
            P2 = conversion['P2']
            P3 = conversion['P3']
            P4 = conversion['P4']
            P5 = conversion['P5']
            P6 = conversion['P6']
            X = values['vals']
            vals = (P1 * X**2 + P2 * X + P3) / (P4 * X**2 + P5 * X + P6)

        elif conversion_type == CONVERSION_TYPE_ALG:
            formula = gp['texts']['conversions'][ch_nr]['formula_addr'].text_str
            X = values['vals']
            vals = evaluate(formula)

        elif conversion_type in (CONVERSION_TYPE_TABI, CONVERSION_TYPE_TAB):
            nr = conversion['val_param_nr'] // 2
            raw = array([conversion['raw_{}'.format(i)] for i in range(nr)])
            phys = array([conversion['phys_{}'.format(i)] for i in range(nr)])
            if conversion_type == CONVERSION_TYPE_TABI:
                vals = interp(values['vals'], raw, phys)
            else:
                idx = searchsorted(raw, values['vals'])
                idx = clip(idx, 0, len(raw) - 1)
                vals = phys[idx]

        elif conversion_type ==  CONVERSION_TYPE_RTAB:
            nr = (conversion['val_param_nr'] - 1) // 3
            lower = array([conversion['lower_{}'.format(i)] for i in range(nr)])
            upper = array([conversion['upper_{}'.format(i)] for i in range(nr)])
            phys = array([conversion['phys_{}'.format(i)] for i in range(nr)])
            default = conversion['default']
            vals = values['vals']

            res = []
            for v in vals:
                for l, u, p in zip(lower, upper, phys):
                    if l <= v <= u:
                        res.append(p)
                        break
                else:
                    res.append(default)
            vals = array(res).astype(ch_fmt)

        elif conversion_type == CONVERSION_TYPE_TABX:
            nr = conversion['val_param_nr']
            raw = array([conversion['val_{}'.format(i)] for i in range(nr)])
            phys = array([gp['texts']['conversion_tab'][ch_nr]['text_{}'.format(i)]['text'] for i in range(nr)])
            default = gp['texts']['conversion_tab'][ch_nr].get('default_addr', {}).get('text', b'')
            vals = values['vals']
            info = {'raw': raw, 'phys': phys, 'default': default, 'type': CONVERSION_TYPE_TABX}

        elif conversion_type == CONVERSION_TYPE_RTABX:
            nr = conversion['val_param_nr'] // 2

            phys = array([gp['texts']['conversion_tab'][ch_nr]['text_{}'.format(i)]['text'] for i in range(nr)])
            lower = array([conversion['lower_{}'.format(i)] for i in range(nr)])
            upper = array([conversion['upper_{}'.format(i)] for i in range(nr)])
            default = gp['texts']['conversion_tab'][ch_nr].get('default_addr', {}).get('text', b'')
            vals = values['vals']
            info = {'lower': lower, 'upper': upper, 'phys': phys, 'default': default, 'type': CONVERSION_TYPE_RTABX}

        elif conversion == CONVERSION_TYPE_TTAB:
            nr = conversion['val_param_nr'] - 1

            raw = array([gp['texts']['conversion_tab'][ch_nr]['text_{}'.format(i)]['text'] for i in range(nr)])
            phys = array([conversion['val_{}'.format(i)] for i in range(nr)])
            default = conversion['val_default']
            vals = values['vals']
            info = {'lower': lower, 'upper': upper, 'phys': phys, 'default': default, 'type': CONVERSION_TYPE_TTAB}

        elif conversion == CONVERSION_TYPE_TRANS:
            nr = (conversion['ref_param_nr'] - 1 ) // 2
            in_ = array([gp['texts']['conversion_tab'][ch_nr]['input_{}'.format(i)]['text'] for i in range(nr)])
            out_ = array([gp['texts']['conversion_tab'][ch_nr]['output_{}'.format(i)]['text'] for i in range(nr)])
            default = gp['texts']['conversion_tab'][ch_nr]['default_addr']['text']
            vals = values['vals']

            res = []
            for v in vals:
                for i, o in zip(in_, out_):
                    if v == i:
                        res.append(o)
                        break
                else:
                    res.append(default)
            vals = array(res)
            info = {'input': in_, 'output': out_, 'default': default, 'type': CONVERSION_TYPE_TRANS}

        if conversion_type in (CONVERSION_TYPE_TABX, CONVERSION_TYPE_RTABX, CONVERSION_TYPE_TTAB, CONVERSION_TYPE_TRANS):
            conversion = info
        else:
            conversion = None

        if return_info:
            return vals, channel.name, conversion, unit
        else:
            return vals

    def get(self, name=None, group=None, index=None, raster=None):
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

            * for TABX conversion:

                * raw - numpy.array for X-axis
                * phys - numpy.array of strings for Y-axis
                * type - conversion type = CONVERSION_TYPE_TABX
                * default - default bytes value

            * for RTABX conversion:

                * lower - numpy.array for lower range
                * upper - numpy.array for upper range
                * phys - numpy.array of strings for Y-axis
                * type - conversion type =
                * default - default bytes value

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
            if not name in self.channels_db:
                raise MdfException('Channel "{}" not found'.format(name))
            else:
                gp_nr, ch_nr = self.channels_db[name]

        gp = self.groups[gp_nr]
        channel = gp['channels'][ch_nr]

        if not self.load_measured_data:
            with open(self.name, 'rb') as file_stream:
                # go to the first data block of the current data group
                dat_addr = gp['data_group']['data_block_addr']
                data = self._read_data_block(address=dat_addr, file_stream=file_stream)

                # check if it is a VLDS channel with signal data
                ch_data_addr = channel['data_block_addr']
                signal_data = self._read_agregated_signal_data(address=ch_data_addr, file_stream=file_stream)
        else:
            if gp['data_block']:
                data = gp['data_block']['data']
            else:
                data = b''
            signal_data = gp['signal_data'][ch_nr]['data'] if gp['signal_data'][ch_nr] else b''

        t = self.get_master_data(group=gp_nr, data=data)

        if ch_nr == self.masters_db[gp_nr]:
            res = Signal(samples=t,
                         timestamps=t[:],
                         unit='s',
                         name=channel.name,
                         conversion=None)
        else:
            vals, name, conversion, unit = self.get_channel_data(group=gp_nr, index=ch_nr, data=data, signal_data=signal_data, return_info=True)

            res = Signal(samples=vals,
                         timestamps=t,
                         unit=unit,
                         name=name,
                         conversion=conversion)

        if raster:
            tx = linspace(0, t[-1], int(t[-1] / raster))
            res = res.interp(tx)
        return res

    def info(self):
        """get MDF information as a dict

        Examples
        --------
        >>> mdf = MDF4('test.mdf')
        >>> mdf.info()


        """
        info = {}
        info['version'] = self.identification['version_str'].strip(b'\x00').decode('utf-8')
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
        >>> mdf = MDF4('test.mdf')
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
                warnings.warn('Group index "{}" not in valid range[0..{}]'.format(group, len(self.groups)))
                return
        elif name:
            if name in self.channels_db:
                idx = self.channels_db[name][1]
            else:
                warnings.warn('Channel name "{}" not found in the measurement'.format(name))
                return
        else:
            warnings.warn('Must specify a valid group or name argument')
            return
        self.groups.pop(idx)

    def save(self, dst=None):
        """Save MDF to *dst*. If *dst* is *None* the original file is overwritten

        """
        if self.load_measured_data == False:
            warnings.warn("Can't save if load_measurement_data option is False")
            return

        if self.name is None and dst is None:
            warnings.warn('New MDF created without a name and no destination file name specified for save')
            return

        dst = dst if dst else self.name

        if not self.file_history:
            comment = 'created'
        else:
            comment = 'updated'

        self.file_history.append([FileHistory(), TextBlock.from_text('<FHcomment>\n<TX>{}</TX>\n<tool_id>PythonMDFEditor</tool_id>\n<tool_vendor></tool_vendor>\n<tool_version>1.0</tool_version>\n</FHcomment>'.format(comment), meta=True)])

        with open(dst, 'wb') as dst:
            defined_texts = {}

            write = dst.write
            tell = dst.tell

            address = IDENTIFICATION_BLOCK_SIZE + HEADER_BLOCK_SIZE
            write(b'\x00' * address)

            if self.file_comment:
                self.file_comment.address = address
                write(bytes(self.file_comment))
                address = tell()

            # write attachemnts
            if self.attachments:
                for at_block, texts in self.attachments:
                    for key, text in texts.items():
                        at_block[key] = text.address = address
                        write(bytes(text))
                        address = tell()

                for at_block, texts in self.attachments:
                    at_block.address = address
                    address += at_block['block_len']
                    align = address % 8
                    if align:
                        address += 8 - align

                for i, (at_block, text) in enumerate(self.attachments[:-1]):
                    at_block['next_at_addr'] = self.attachments[i+1][0].address
                self.attachments[-1][0]['next_at_addr'] = 0

                for at_block, texts in self.attachments:
                    write(bytes(at_block))
                    address = tell()
                    align = address % 8
                    if align:
                        write(b'\x00' * (8 - align))
                        address += 8 - align

            # write file history blocks
            for i, (fh, fh_text) in enumerate(self.file_history):
                fh_text.address = address
                write(bytes(fh_text))
                address = tell()

                fh['comment_addr'] = fh_text.address

            for i, (fh, fh_text) in enumerate(self.file_history):
                fh.address = address
                address += FH_BLOCK_SIZE

            for i, (fh, fh_text) in enumerate(self.file_history[:-1]):
                fh['next_fh_addr'] = self.file_history[i+1][0].address
            self.file_history[-1][0]['next_fh_addr'] = 0
            for fh, _ in self.file_history:
                write(bytes(fh))
            address = tell()

            for i, gp in enumerate(self.groups):
                # write TXBLOCK's
                for _, item_list in gp['texts'].items():
                    for dict_ in item_list:
                        for key in dict_:
                            #text blocks can be shared
                            if dict_[key].text_str in defined_texts:
                                dict_[key].address = defined_texts[dict_[key].text_str]
                            else:
                                defined_texts[dict_[key].text_str] = address
                                dict_[key].address = address
                                write(bytes(dict_[key]))
                                address = tell()

                # write channel conversions
                for j, conv in enumerate(gp['channel_conversions']):
                    if conv:
                        conv.address = address

                        for key in ('name_addr', 'unit_addr', 'comment_addr', 'formula_addr'):
                            if key in gp['texts']['conversions'][j]:
                                conv[key] = gp['texts']['conversions'][j][key].address
                            else:
                                conv[key] = 0
                        conv['inv_conv_addr'] = 0

                        if conv['conversion_type'] in (CONVERSION_TYPE_TABX,
                                                       CONVERSION_TYPE_RTABX,
                                                       CONVERSION_TYPE_TTAB,
                                                       CONVERSION_TYPE_TRANS):
                            for key in gp['texts']['conversion_tab'][j]:
                                conv[key] = gp['texts']['conversion_tab'][j][key].address

                        write(bytes(conv))
                        address = tell()

                for j, source in enumerate(gp['channel_sources']):
                    if source:
                        source.address = address

                        for key in ('name_addr', 'path_addr', 'comment_addr'):
                            if key in gp['texts']['sources'][j]:
                                source[key] = gp['texts']['sources'][j][key].address
                            else:
                                source[key] = 0

                        write(bytes(source))
                        address = tell()

                for j, signal_data in enumerate(gp['signal_data']):
                    if signal_data:
                        signal_data.address = address
                        write(bytes(signal_data))
                        address = tell()

                for j, (channel, signal_data) in enumerate(zip(gp['channels'], gp['signal_data'])):
                    channel.address = address
                    address += CN_BLOCK_SIZE

                    for key in ('name_addr', 'comment_addr', 'unit_addr'):
                        if key in gp['texts']['channels'][j]:
                            channel[key] = gp['texts']['channels'][j][key].address
                        else:
                            channel[key] = 0
                    channel['conversion_addr'] = 0 if not gp['channel_conversions'][j] else gp['channel_conversions'][j].address
                    channel['source_addr'] = gp['channel_sources'][j].address if gp['channel_sources'][j] else 0
                    channel['data_block_addr'] = signal_data.address if signal_data else 0

                for channel, next_channel in pair(gp['channels']):
                    channel['next_ch_addr'] = next_channel.address
                    write(bytes(channel))
                next_channel['next_ch_addr'] = 0
                write(bytes(next_channel))
                address = tell()

                gp['channel_group'].address = address
                gp['channel_group']['first_ch_addr'] = gp['channels'][0].address
                gp['channel_group']['next_cg_addr'] = 0
                for key in ('acq_name_addr', 'comment_addr'):
                    if key in gp['texts']['channel_group'][0]:
                        gp['channel_group'][key] = gp['texts']['channel_group'][0][key].address
                gp['channel_group']['acq_source_addr'] = 0
                write(bytes(gp['channel_group']))
                address = tell()

                #print(len(self.groups), self.groups.index(gp))

                if gp['data_block']:
                    block = gp['data_block']

                    block.address = address
                    address += block['block_len']
                    align = address % 8
                    if align:
                        add = 8 - align
                        address += add
                    else:
                        add = 0
                    write(bytes(block) + b'\x00' * add)
                    address = tell()

            for gp in self.groups:
                gp['data_group'].address = address
                address += DG_BLOCK_SIZE

                gp['data_group']['first_cg_addr'] = gp['channel_group'].address
                gp['data_group']['comment_addr'] = 0
                if gp['data_block']:
                    gp['data_group']['data_block_addr'] = gp['data_block'].address
                else:
                    gp['data_group']['data_block_addr'] = 0

            for i, dg in enumerate(self.groups[:-1]):
                dg['data_group']['next_dg_addr'] = self.groups[i+1]['data_group'].address
            self.groups[-1]['data_group']['next_dg_addr'] = 0

            for dg in (dg_['data_group'] for dg_ in self.groups):
                write(bytes(dg))

            if self.groups:
                self.header['first_dg_addr'] = self.groups[0]['data_group'].address
            else:
                self.header['first_dg_addr'] = 0
            self.header['file_history_addr'] = self.file_history[0][0].address
            self.header['first_attachment_addr'] = self.attachments[0][0].address if self.attachments else 0
            self.header['comment_addr'] = self.file_comment.address if self.file_comment else 0
            dst.seek(0, SEEK_START)
            write(bytes(self.identification))
            write(bytes(self.header))


if __name__ == '__main__':
    pass

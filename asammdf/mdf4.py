# -*- coding: utf-8 -*-
"""
ASAM MDF version 4 file format module
"""
from __future__ import print_function, division
import sys
import time
import warnings
import os

from tempfile import TemporaryFile
from struct import unpack, unpack_from
from functools import reduce, partial
from collections import defaultdict
from hashlib import md5
import xml.etree.ElementTree as XML

from numpy import (interp, linspace, dtype, array_equal, zeros, uint8,
                   array, searchsorted, clip, union1d, float64, frombuffer,
                   argwhere, arange, flip, unpackbits, packbits, roll,
                   transpose, issubdtype, unsignedinteger, integer, signedinteger)
from numpy.core.records import fromstring, fromarrays
from numpy.core.defchararray import encode
from numexpr import evaluate

from .v4blocks import (AttachmentBlock,
                       Channel,
                       ChannelArrayBlock,
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

from . import v4constants as v4c
from .utils import (MdfException,
                    get_fmt,
                    fmt_to_datatype,
                    pair,
                    get_unique_name,
                    get_min_max,
                    fix_dtype_fields)

from .signal import Signal


get_fmt = partial(get_fmt, version=4)
fmt_to_datatype = partial(fmt_to_datatype, version=4)

MASTER_CHANNELS = (v4c.CHANNEL_TYPE_MASTER, v4c.CHANNEL_TYPE_VIRTUAL_MASTER)

PYVERSION = sys.version_info[0]
if PYVERSION == 2:
    from .utils import bytes

__all__ = ['MDF4', ]


class MDF4(object):
    """If the *name* exist it will be loaded otherwise an empty file will be
    created that can be later saved to disk

    Parameters
    ----------
    name : string
        mdf file name
    load_measured_data : bool
        load data option; default *True*

        * if *True* the data group binary data block will be loaded in RAM
        * if *False* the channel data is read from disk on request

    version : string
        mdf file version ('4.00', '4.10', '4.11'); default '4.10'


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
    version : str
        mdf version
    channels_db : dict
        used for fast channel access by name; for each name key the value is a
        list of (group index, channel index) tuples
    masters_db : dict
        used for fast master channel access; for each group index key the value
         is the master channel index

    """

    _compact_integers_on_append = False

    def __init__(self, name=None, load_measured_data=True, version='4.10'):
        self.groups = []
        self.header = None
        self.identification = None
        self.file_history = []
        self.file_comment = None
        self.name = name
        self.load_measured_data = load_measured_data
        self.channels_db = {}
        self.masters_db = {}
        self.attachments = []

        self._ch_map = {}
        self._master_channel_cache = {}

        # used for appending when load_measured_data=False
        self._tempfile = None

        if name:
            with open(self.name, 'rb') as file_stream:
                self._read(file_stream)

        else:
            self.header = HeaderBlock()
            self.identification = FileIdentificationBlock(version=version)
            self.version = version

    @classmethod
    def _enable_integer_compacting(cls, enable):
        """ enable or disable compacting of integer channels when appending

        Parameters
        ----------
        enable : bool

        """

        cls._compact_integers_on_append = enable

    def _check_finalised(self):
        unfinalised_stdandard_flags = self.identification['unfinalized_standard_flags']
        if unfinalised_stdandard_flags & 1:
            message = ('Unfinalised file {}:'
                       'Update of cycle counters for CG/CA blocks required')
            warnings.warn(message.format(self.name))
        elif unfinalised_stdandard_flags & 1 << 1:
            message = ('Unfinalised file {}:'
                       'Update of cycle counters for SR blocks required')
            warnings.warn(message.format(self.name))
        elif unfinalised_stdandard_flags & 1 << 2:
            message = ('Unfinalised file {}:'
                       'Update of length for last DT block required')
            warnings.warn(message.format(self.name))
        elif unfinalised_stdandard_flags & 1 << 3:
            message = ('Unfinalised file {}:'
                       'Update of length for last RD block required')
            warnings.warn(message.format(self.name))
        elif unfinalised_stdandard_flags & 1 << 4:
            message = ('Unfinalised file {}:'
                       'Update of last DL block in each chained list'
                       'of DL blocks required')
            warnings.warn(message.format(self.name))
        elif unfinalised_stdandard_flags & 1 << 5:
            message = ('Unfinalised file {}:'
                       'Update of cg_data_bytes and cg_inval_bytes '
                       'in VLSD CG block required')
            warnings.warn(message.format(self.name))
        elif unfinalised_stdandard_flags & 1 << 6:
            message = ('Unfinalised file {}:'
                       'Update of offset values for VLSD channel required '
                       'in case a VLSD CG block is used')
            warnings.warn(message.format(self.name))

    def _read(self, file_stream):
        dg_cntr = 0

        self.identification = FileIdentificationBlock(file_stream=file_stream)
        self.version = self.identification['version_str'].decode('utf-8')\
                                                         .strip(' \n\t\x00')

        if self.version == '4.10':
            self._check_finalised()

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
                grp['data_block'] = None
                grp['channel_dependencies'] = []
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

                dg_cntr += 1

            # store channel groups record sizes dict in each
            # new group data belong to the initial unsorted group, and add
            # the key 'sorted' with the value False to use a flag;
            # this is used later if load_measured_data=False

            if cg_nr > 1:
                cg_size = {}
                for grp in new_groups:
                    if grp['channel_group']['flags'] == 0:
                        cg_size[grp['channel_group']['record_id']] = grp['channel_group']['samples_byte_nr'] + grp['channel_group']['invalidation_bytes_nr']
                    else:
                        # VLDS flags
                        cg_size[grp['channel_group']['record_id']] = 0

                for grp in new_groups:
                    grp['sorted'] = False
                    grp['record_size'] = cg_size
            else:
                grp['sorted'] = True

            if self.load_measured_data:
                # go to the first data block of the current data group
                dat_addr = group['data_block_addr']
                data = self._read_data_block(address=dat_addr, file_stream=file_stream)

                if cg_nr == 1:
                    grp = new_groups[0]
                    grp['data_location'] = v4c.LOCATION_MEMORY
                    grp['data_block'] = DataBlock(data=data)
                else:
                    cg_data = defaultdict(list)
                    record_id_nr = group['record_id_len'] if group['record_id_len'] <= 2 else 0
                    i = 0
                    size = len(data)
                    while i < size:
                        rec_id = data[i]
                        # skip record id
                        i += 1
                        rec_size = cg_size[rec_id]
                        if rec_size:
                            rec_data = data[i: i+rec_size]
                            cg_data[rec_id].append(rec_data)
                        else:
                            # as shown bby mdfvalidator rec size is first byte after rec id + 3
                            rec_size = unpack('<I', data[i: i+4])[0]
                            i += 4
                            rec_data = data[i: i + rec_size]
                            cg_data[rec_id].append(rec_data)
                        # if 2 record id's are used skip also the second one
                        if record_id_nr == 2:
                            i += 1
                        # go to next record
                        i += rec_size
                    for grp in new_groups:
                        grp['data_location'] = v4c.LOCATION_MEMORY
                        kargs = {}
                        kargs['data'] = b''.join(cg_data[grp['channel_group']['record_id']])
                        grp['channel_group']['record_id'] = 1
                        grp['data_block'] = DataBlock(**kargs)
            else:
                for grp in new_groups:
                    grp['data_location'] = v4c.LOCATION_ORIGINAL_FILE

            self.groups.extend(new_groups)

            dg_addr = group['next_dg_addr']

        for grp in self.groups:
            for dependency_list in grp['channel_dependencies']:
                if dependency_list:
                    for dep in dependency_list:
                        if isinstance(dep, Channel):
                            break
                        else:
                            if dep['ca_type'] == v4c.CA_TYPE_LOOKUP and dep['links_nr'] == 4 * dep['dims'] + 1:
                                for i in range(dep['dims']):
                                    ch_addr = dep['scale_axis_{}_ch_addr'.format(i)]
                                    dep.referenced_channels.append(self._ch_map[ch_addr])

    def _read_channels(self, ch_addr, grp, file_stream, dg_cntr, ch_cntr, channel_composition=False):

        channels = grp['channels']
        composition = []
        while ch_addr:
            # read channel block and create channel object
            channel = Channel(address=ch_addr, file_stream=file_stream)

            self._ch_map[ch_addr] = (ch_cntr, dg_cntr)

            channels.append(channel)
            if channel_composition:
                composition.append(channel)

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
            # read text fields for channel conversions
            conv_texts = {}
            grp['texts']['conversions'].append(conv_texts)

            if conv:
                for key in ('name_addr', 'unit_addr', 'comment_addr'):
                    address = conv[key]
                    if address:
                        conv_texts[key] = TextBlock(address=address, file_stream=file_stream)
                if 'formula_addr' in conv:
                    address = conv['formula_addr']
                    if address:
                        conv_texts['formula_addr'] = TextBlock(address=address, file_stream=file_stream)
                    else:
                        conv_texts['formula_addr'] = None

                if conv['conversion_type'] in v4c.TABULAR_CONVERSIONS:
                    # link_nr - common links (4) - default text link (1)
                    for i in range(conv['links_nr'] - 4 - 1):
                        address = conv['text_{}'.format(i)]
                        if address:
                            conv_tabx_texts['text_{}'.format(i)] = TextBlock(address=address, file_stream=file_stream)
                    address = conv.get('default_addr', 0)
                    if address:
                        file_stream.seek(address, v4c.SEEK_START)
                        blk_id = file_stream.read(4)
                        if blk_id == b'##TX':
                            conv_tabx_texts['default_addr'] = TextBlock(address=address, file_stream=file_stream)
                        elif blk_id == b'##CC':
                            conv_tabx_texts['default_addr'] = ChannelConversion(address=address, file_stream=file_stream)
                            conv_tabx_texts['default_addr']['text'] = str(time.clock()).encode('utf-8')

                            conv['unit_addr'] = conv_tabx_texts['default_addr']['unit_addr']
                            conv_tabx_texts['default_addr']['unit_addr'] = 0
                elif conv['conversion_type'] == v4c.CONVERSION_TYPE_TRANS:
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

            # read text fields for channel
            channel_texts = {}
            grp['texts']['channels'].append(channel_texts)
            for key in ('name_addr', 'comment_addr', 'unit_addr'):
                address = channel[key]
                if address:
                    channel_texts[key] = TextBlock(address=address, file_stream=file_stream)

            # update channel object name and block_size attributes
            channel.name = channel_texts['name_addr']['text'].decode('utf-8').strip(' \t\n\r\x00')
            if channel.name not in self.channels_db:
                self.channels_db[channel.name] = []
            self.channels_db[channel.name].append((dg_cntr, ch_cntr))

            if channel['channel_type'] in MASTER_CHANNELS:
                self.masters_db[dg_cntr] = ch_cntr

            ch_cntr += 1

            if channel['component_addr']:
                # check if it is a CABLOCK or CNBLOCK
                file_stream.seek(channel['component_addr'], v4c.SEEK_START)
                blk_id = file_stream.read(4)
                if blk_id == b'##CN':
                    ch_cntr, composition = self._read_channels(channel['component_addr'], grp, file_stream, dg_cntr, ch_cntr, True)
                    grp['channel_dependencies'].append(composition)
                else:
                    # only channel arrays with storage=CN_TEMPLATE are supported so far
                    ca_block = ChannelArrayBlock(address=channel['component_addr'], file_stream=file_stream)
                    if ca_block['storage'] != v4c.CA_STORAGE_TYPE_CN_TEMPLATE:
                        warnings.warn('Only CN template arrays are supported')
                    ca_list = [ca_block, ]
                    while ca_block['composition_addr']:
                        ca_block = ChannelArrayBlock(address=ca_block['composition_addr'], file_stream=file_stream)
                        ca_list.append(ca_block)
                    grp['channel_dependencies'].append(ca_list)

            else:
                grp['channel_dependencies'].append(None)

            # go to next channel of the current channel group
            ch_addr = channel['next_ch_addr']

        return ch_cntr, composition

    def _read_data_block(self, address, file_stream):
        """read and agregate data blocks for a given data group

        Returns
        -------
        data : bytes
            agregated raw data
        """
        if address:
            file_stream.seek(address, v4c.SEEK_START)
            id_string = file_stream.read(4)
            # can be a DataBlock
            if id_string == b'##DT':
                data = DataBlock(address=address, file_stream=file_stream)['data']
            # or a DataZippedBlock
            elif id_string == b'##DZ':
                data = DataZippedBlock(address=address, file_stream=file_stream)['data']
            # or a DataList
            elif id_string == b'##DL':
                data = []
                while address:
                    dl = DataList(address=address, file_stream=file_stream)
                    for i in range(dl['links_nr'] - 1):
                        addr = dl['data_block_addr{}'.format(i)]
                        file_stream.seek(addr, v4c.SEEK_START)
                        id_string = file_stream.read(4)
                        if id_string == b'##DT':
                            data.append(DataBlock(file_stream=file_stream, address=addr)['data'])
                        elif id_string == b'##DZ':
                            data.append(DataZippedBlock(address=addr, file_stream=file_stream)['data'])
                        elif id_string == b'##DL':
                            data.append(self._read_data_block(address=addr, file_stream=file_stream))
                    address = dl['next_dl_addr']
                data = b''.join(data)
            # or a header list
            elif id_string == b'##HL':
                hl = HeaderList(address=address, file_stream=file_stream)
                return self._read_data_block(address=hl['first_dl_addr'], file_stream=file_stream)
        else:
            data = b''
        return data

    def _load_group_data(self, group):
        """ get group's data block bytes """
        if self.load_measured_data == False:
            # could be an appended group
            # for now appended groups keep the measured data in the memory.
            # the plan is to use a temp file for appended groups, to keep the
            # memory usage low.
            if group['data_location'] == v4c.LOCATION_ORIGINAL_FILE:
                with open(self.name, 'rb') as file_stream:
                    # go to the first data block of the current data group
                    dat_addr = group['data_group']['data_block_addr']
                    data = self._read_data_block(address=dat_addr, file_stream=file_stream)

                    if not group['sorted']:
                        cg_data = []
                        cg_size = group['record_size']
                        record_id = group['channel_group']['record_id']
                        record_id_nr = group['data_group']['record_id_len'] if group['data_group']['record_id_len'] <= 2 else 0
                        i = 0
                        size = len(data)
                        while i < size:
                            rec_id = data[i]
                            # skip record id
                            i += 1
                            rec_size = cg_size[rec_id]
                            if rec_size:
                                if rec_id == record_id:
                                    rec_data = data[i: i+rec_size]
                                    cg_data.append(rec_data)
                            else:
                                # as shown bby mdfvalidator rec size is first byte after rec id + 3
                                rec_size = unpack('<I', data[i: i+4])[0]
                                i += 4
                                if rec_id == record_id:
                                    rec_data = data[i: i + rec_size]
                                    cg_data.append(rec_data)
                            # if 2 record id's are used skip also the second one
                            if record_id_nr == 2:
                                i += 1
                            # go to next record
                            i += rec_size
                        data = b''.join(cg_data)
            elif group['data_location'] == v4c.LOCATION_TEMPORARY_FILE:
                dat_addr = group['data_group']['data_block_addr']
                self._tempfile.seek(dat_addr, v4c.SEEK_START)
                size = group['channel_group']['samples_byte_nr'] * group['channel_group']['cycles_nr']
                data = self._tempfile.read(size)
        else:
            data = group['data_block']['data']

        return data

    def _read_agregated_signal_data(self, address, file_stream):
        """ this method is used to get the channel signal data, usually for VLSD channels """
        if address:
            file_stream.seek(address, v4c.SEEK_START)
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
                        file_stream.seek(addr, v4c.SEEK_START)
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
            elif blk_id == b'##CN':
                data = b''
            else:
                warnings.warn('Expected SD, DL, DZ or CN block at {} but found id="{}"'.format(hex(address), blk_id))
                return
        else:
            data = b''

        return data

    def _prepare_record(self, group):
        """ compute record dtype and parents dict fro this group

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
        record_size = grp['channel_group']['samples_byte_nr']
        invalidation_bytes_nr = grp['channel_group']['invalidation_bytes_nr']
        next_byte_aligned_position = 0
        types = []
        current_parent = ""
        parent_start_offset = 0
        parents = {}
        group_channels = set()

        # the channels are first sorted ascending (see __lt__ method of Channel class):
        # a channel with lower byte offset is smaller, when two channels have
        # the same byte offset the one with higer bit size + bit offset is considered smaller, and
        # if bit size + bit offset are equal then the one with lower bit_offset is smaller.
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

            start_offset = new_ch['byte_offset']
            bit_offset = new_ch['bit_offset']
            data_type = new_ch['data_type']
            bit_count = new_ch['bit_count']
            ch_type = new_ch['channel_type']
            dependency_list = grp['channel_dependencies'][original_index]
            name = new_ch.name

            # handle multiple occurance of same channel name
            name = get_unique_name(group_channels, name)
            group_channels.add(name)

            if start_offset >= next_byte_aligned_position:
                if ch_type not in (v4c.CHANNEL_TYPE_VIRTUAL_MASTER, v4c.CHANNEL_TYPE_VIRTUAL):
                    if not dependency_list:
                        parent_start_offset = start_offset

                        # check if there are byte gaps in the record
                        gap = parent_start_offset - next_byte_aligned_position
                        if gap:
                            types.append( ('', 'a{}'.format(gap)) )

                        # adjust size to 1, 2, 4 or 8 bytes
                        size = bit_offset + bit_count
                        if not data_type in (v4c.DATA_TYPE_BYTEARRAY,
                                             v4c.DATA_TYPE_STRING_UTF_8,
                                             v4c.DATA_TYPE_STRING_LATIN_1,
                                             v4c.DATA_TYPE_STRING_UTF_16_BE,
                                             v4c.DATA_TYPE_STRING_UTF_16_LE,
                                             v4c.DATA_TYPE_CANOPEN_TIME,
                                             v4c.DATA_TYPE_CANOPEN_DATE):
                            if size > 32:
                                size = 8
                            elif size > 16:
                                size = 4
                            elif size > 8:
                                size = 2
                            else:
                                size = 1
                        else:
                            size = size >> 3

                        next_byte_aligned_position = parent_start_offset + size
                        if next_byte_aligned_position <= record_size:
                            types.append( (name, get_fmt(data_type, size)) )
                            parents[original_index] = name, bit_offset

                        current_parent = name
                    else:
                        if isinstance(dependency_list[0], ChannelArrayBlock):
                            ca_block = dependency_list[0]

                            # assume that the channel array is byte aligned

                            # check if there are byte gaps in the record
                            gap = start_offset - next_byte_aligned_position
                            if gap:
                                types.append( ('', 'a{}'.format(gap)) )

                            size = bit_count >> 3
                            shape = tuple(ca_block['dim_size_{}'.format(i)] for i in range(ca_block['dims']))
                            if ca_block['byte_offset_base'] // size > 1 and len(shape) == 1:
                                shape += (ca_block['byte_offset_base'] // size, )
                            dim = 1
                            for d in shape:
                                dim *= d

                            types.append( (name, get_fmt(data_type, size), shape) )

                            current_parent = name
                            next_byte_aligned_position = start_offset + size * dim
                            parents[original_index] = name, 0

                        else:
                            parents[original_index] = None, None
                # virtual channels do not have bytes in the record
                else:
                    parents[original_index] = None, None

            else:
                max_overlapping_size = (next_byte_aligned_position - start_offset) * 8
                needed_size = bit_offset + bit_count
                if max_overlapping_size >= needed_size:
                    parents[original_index] = current_parent, ((start_offset - parent_start_offset) << 3 ) + bit_offset
            if next_byte_aligned_position > record_size:
                break

        gap = record_size - next_byte_aligned_position
        if gap > 0:
            types.append( ('', 'a{}'.format(gap)) )

        types.append( ('invalidation_bytes', 'a{}'.format(invalidation_bytes_nr)) )
        if PYVERSION == 2:
            types = fix_dtype_fields(types)

        return parents, dtype(types)

    def _get_not_byte_aligned_data(self, data, group, ch_nr):
        big_endian_types = (v4c.DATA_TYPE_UNSIGNED_MOTOROLA,
                            v4c.DATA_TYPE_REAL_MOTOROLA,
                            v4c.DATA_TYPE_SIGNED_MOTOROLA)

        record_size = group['channel_group']['samples_byte_nr']

        channel = group['channels'][ch_nr]

        bit_offset = channel['bit_offset']
        byte_offset = channel['byte_offset']
        bit_count = channel['bit_count']

        dependency_list = group['channel_dependencies'][ch_nr]
        if dependency_list and isinstance(dependency_list[0], ChannelArrayBlock):
            ca_block = dependency_list[0]

            size = bit_count >> 3
            shape = tuple(ca_block['dim_size_{}'.format(i)] for i in range(ca_block['dims']))
            if ca_block['byte_offset_base'] // size > 1 and len(shape) == 1:
                shape += (ca_block['byte_offset_base'] // size, )
            dim = 1
            for d in shape:
                dim *= d
            size *= dim
            bit_count = size << 3

        byte_count = bit_offset + bit_count
        if byte_count % 8:
            byte_count = (byte_count >> 3) + 1
        else:
            byte_count >>= 3

        types = [('', 'a{}'.format(byte_offset)),
                 ('vals', '({},)u1'.format(byte_count)),
                 ('', 'a{}'.format(record_size - byte_count - byte_offset))]

        vals = fromstring(data, dtype=dtype(types))

        vals = vals['vals']

        if channel['data_type'] not in big_endian_types:
            vals = flip(vals, 1)

        vals = unpackbits(vals)
        vals = roll(vals, bit_offset)
        vals = vals.reshape((len(vals) // 8, 8))
        vals = packbits(vals)
        vals = vals.reshape((len(vals) // byte_count, byte_count))

        if bit_count < 64:
            mask = 2 ** bit_count - 1
            masks = []
            while mask:
                masks.append(mask & 0xFF)
                mask >>= 8
            for i in range(byte_count - len(masks)):
                masks.append(0)

            masks = masks[::-1]
            for i, mask in enumerate(masks):
                vals[:, i] &= mask

        if channel['data_type'] not in big_endian_types:
            vals = flip(vals, 1)

        if bit_count <= 8:
            size = 1
        elif bit_count <= 16:
            size = 2
        elif bit_count <= 32:
            size = 4
        elif bit_count <= 64:
            size = 8

        if size > byte_count:
            extra_bytes = size - byte_count
            extra = zeros((len(vals), extra_bytes), dtype=uint8)

            types = [('vals', vals.dtype, vals.shape[1:]),
                      ('', extra.dtype, extra.shape[1:])]
            vals = fromarrays([vals, extra], dtype=dtype(types))
        vals = vals.tostring()

        fmt = get_fmt(channel['data_type'], size)
        if size <= byte_count:
            types = [('vals', fmt),
                     ('', 'a{}'.format(byte_count - size))]
        else:
            types = [('vals', fmt),]

        vals = fromstring(vals, dtype=dtype(types))

        return vals['vals']

    def append(self, signals, source_info='Python', common_timebase=False, compact=True):
        """
        Appends a new data group.

        For channel depencies type Signals, the *samples* attribute must be a numpy.recarray

        Parameters
        ----------
        signals : list
            list on *Signal* objects
        source_info : str
            source information; default 'Python'
        common_timebase : bool
            flag to hint that the signals have the same timebase
        compact : bool
            compact unsigned signals if possible; this can decrease the file
            size but increases the execution time

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

        # split regular from composed signals. Composed signals have recarray samples
        # or multimendional ndarray.
        # The regular signals will be first added to the group.
        # The composed signals will be saved along side the fields, which will
        # be saved as new signals.
        simple_signals = [sig for sig in signals
                          if len(sig.samples.shape) <= 1
                          and sig.samples.dtype.names is None]
        composed_signals = [sig for sig in signals
                            if len(sig.samples.shape) > 1
                            or sig.samples.dtype.names]

        dg_cntr = len(self.groups)

        gp = {}
        gp['channels'] = gp_channels = []
        gp['channel_conversions'] = gp_conv = []
        gp['channel_sources'] = gp_source = []
        gp['channel_dependencies'] = gp_dep = []
        gp['texts'] = gp_texts = {'channels': [],
                                  'sources': [],
                                  'conversions': [],
                                  'conversion_tab': [],
                                  'channel_group': []}
        gp['signal_data'] = []

        self.groups.append(gp)

        cycles_nr = len(t)
        fields = []
        types = []
        parents = {}
        ch_cntr = 0
        offset = 0
        field_names = set()

        # setup all blocks related to the time master channel

        # time channel texts
        for _, item in gp_texts.items():
            item.append({})
        gp_texts['channels'][-1]['name_addr'] = TextBlock(text='t', meta=False)
        gp_texts['conversions'][-1]['unit_addr'] = TextBlock(text='s', meta=False)

        si_text = TextBlock(text=source_info, meta=False)
        gp_texts['sources'][-1]['name_addr'] = si_text
        gp_texts['sources'][-1]['path_addr'] = si_text
        gp_texts['channel_group'][-1]['acq_name_addr'] = si_text
        gp_texts['channel_group'][-1]['comment_addr'] = si_text

        #conversion for time channel
        gp_conv.append(None)

        #source for time
        gp_source.append(SourceInformation())

        #time channel
        t_type, t_size = fmt_to_datatype(t.dtype)
        kargs = {'channel_type': v4c.CHANNEL_TYPE_MASTER,
                 'data_type': t_type,
                 'sync_type': 1,
                 'byte_offset': 0,
                 'bit_offset': 0,
                 'bit_count': t_size,
                 'min_raw_value': t[0] if cycles_nr else 0,
                 'max_raw_value' : t[-1] if cycles_nr else 0,
                 'lower_limit' : t[0] if cycles_nr else 0,
                 'upper_limit' : t[-1] if cycles_nr else 0,
                 'flags': v4c.FLAG_PHY_RANGE_OK | v4c.FLAG_VAL_RANGE_OK}
        ch = Channel(**kargs)
        ch.name = name = 't'
        gp_channels.append(ch)

        gp['signal_data'].append(None)

        if not name in self.channels_db:
            self.channels_db[name] = []
        self.channels_db[name].append((dg_cntr, ch_cntr))
        self.masters_db[dg_cntr] = 0
        # data group record parents
        parents[ch_cntr] = name, 0

        # time channel doesn't have channel dependencies
        gp_dep.append(None)

        fields.append(t)
        types.append( (name, t.dtype))
        field_names.add(name)

        offset += t_size // 8
        ch_cntr += 1

        if self._compact_integers_on_append:
            compacted_signals = [{'signal': sig} for sig in simple_signals if issubdtype(sig.samples.dtype, integer)]

            max_itemsize = 1
            dtype_ = dtype(uint8)

            for signal in compacted_signals:
                itemsize = signal['signal'].samples.dtype.itemsize

                signal['min'], signal['max'] = get_min_max(signal['signal'].samples)
                minimum_bitlength = (itemsize // 2) * 8 + 1
                bit_length = max(int(signal['max']).bit_length(),
                                 int(signal['min']).bit_length())

                signal['bit_count'] = max(minimum_bitlength, bit_length)

                if itemsize > max_itemsize:
                    dtype_ = dtype('<u{}'.format(itemsize))
                    max_itemsize = itemsize

            compacted_signals.sort(key=lambda x: x['bit_count'])
            simple_signals = [sig for sig in simple_signals if not issubdtype(sig.samples.dtype, integer)]
            dtype_size = dtype_.itemsize * 8

        else:
            compacted_signals = []

        # first try to compact unsigned integers
        while compacted_signals:
            # channels texts

            cluster = []

            tail = compacted_signals.pop()
            size = tail['bit_count']
            cluster.append(tail)

            while size < dtype_size and compacted_signals:
                head = compacted_signals[0]
                head_size = head['bit_count']
                if head_size + size > dtype_size:
                    break
                else:
                    cluster.append(compacted_signals.pop(0))
                    size += head_size

            bit_offset = 0
            field_name = get_unique_name(field_names, 'COMPACT')
            types.append( (field_name, dtype_) )
            field_names.add(field_name)

            values = zeros(cycles_nr, dtype=dtype_)

            for signal_d in cluster:

                signal = signal_d['signal']
                bit_count = signal_d['bit_count']
                min_val = signal_d['min']
                max_val = signal_d['max']

                name = signal.name
                for _, item in gp['texts'].items():
                    item.append({})
                gp_texts['channels'][-1]['name_addr'] = TextBlock(text=name, meta=False)
                if signal.unit:
                    gp_texts['channels'][-1]['unit_addr'] = TextBlock(text=signal.unit, meta=False)
                gp_texts['sources'][-1]['name_addr'] = si_text
                gp_texts['sources'][-1]['path_addr'] = si_text

                # conversions for channel

                info = signal.info
                conv_texts_tab = gp_texts['conversion_tab'][-1]
                if info and 'raw' in info:
                    kargs = {}
                    kargs['conversion_type'] = v4c.CONVERSION_TYPE_TABX
                    raw = info['raw']
                    phys = info['phys']
                    for i, (r_, p_) in enumerate(zip(raw, phys)):
                        kargs['text_{}'.format(i)] = 0
                        kargs['val_{}'.format(i)] = r_
                        conv_texts_tab['text_{}'.format(i)] = TextBlock(text=p_, meta=False)
                    if info.get('default', b''):
                        conv_texts_tab['default_addr'] = TextBlock(text=info['default'], meta=False)
                    kargs['default_addr'] = 0
                    kargs['links_nr'] = len(raw) + 5
                    gp_conv.append(ChannelConversion(**kargs))
                elif info and 'lower' in info:
                    kargs = {}
                    kargs['conversion_type'] = v4c.CONVERSION_TYPE_RTABX
                    lower = info['lower']
                    upper = info['upper']
                    texts = info['phys']
                    kargs['ref_param_nr'] = len(upper)
                    kargs['default_addr'] = info.get('default', 0)
                    kargs['links_nr'] = len(lower) + 5

                    for i, (u_, l_, t_) in enumerate(zip(upper, lower, texts)):
                        kargs['lower_{}'.format(i)] = l_
                        kargs['upper_{}'.format(i)] = u_
                        kargs['text_{}'.format(i)] = 0
                        conv_texts_tab['text_{}'.format(i)] = TextBlock(text=t_, meta=False)
                    if info.get('default', b''):
                        conv_texts_tab['default_addr'] = TextBlock(text=info['default'], meta=False)
                    kargs['default_addr'] = 0
                    gp_conv.append(ChannelConversion(**kargs))

                else:
                    gp_conv.append(None)

                # source for channel
                gp_source.append(SourceInformation())

                # compute additional byte offset for large records size
                kargs = {'channel_type': v4c.CHANNEL_TYPE_VALUE,
                         'bit_count': bit_count,
                         'byte_offset': offset + bit_offset // 8,
                         'bit_offset' : bit_offset % 8,
                         'data_type': v4c.DATA_TYPE_UNSIGNED_INTEL if issubdtype(signal.samples.dtype, unsignedinteger) else v4c.DATA_TYPE_SIGNED_INTEL,
                         'min_raw_value': min_val if min_val<=max_val else 0,
                         'max_raw_value' : max_val if min_val<=max_val else 0,
                         'lower_limit' : min_val if min_val<=max_val else 0,
                         'upper_limit' : max_val if min_val<=max_val else 0}
                if min_val > max_val:
                    kargs['flags'] = 0
                else:
                    kargs['flags'] = v4c.FLAG_PHY_RANGE_OK | v4c.FLAG_VAL_RANGE_OK
                ch = Channel(**kargs)
                ch.name = name
                gp_channels.append(ch)
                gp['signal_data'].append(None)

                if name not in self.channels_db:
                    self.channels_db[name] = []
                self.channels_db[name].append((dg_cntr, ch_cntr))

                # update the parents as well

                parents[ch_cntr] = field_name, bit_offset

                values += signal.samples.astype(dtype_) << bit_offset
                bit_offset += bit_count

                ch_cntr += 1

                # simple channels don't have channel dependencies
                gp_dep.append(None)

            offset += dtype_.itemsize
            fields.append(values)

        # first add the signals in the simple signal list
        for signal in simple_signals:
            # channels texts

            name = signal.name
            for _, item in gp['texts'].items():
                item.append({})
            gp_texts['channels'][-1]['name_addr'] = TextBlock(text=name, meta=False)
            if signal.unit:
                gp_texts['channels'][-1]['unit_addr'] = TextBlock(text=signal.unit, meta=False)
            gp_texts['sources'][-1]['name_addr'] = si_text
            gp_texts['sources'][-1]['path_addr'] = si_text

            # conversions for channel

            info = signal.info
            conv_texts_tab = gp_texts['conversion_tab'][-1]
            if info and 'raw' in info:
                kargs = {}
                kargs['conversion_type'] = v4c.CONVERSION_TYPE_TABX
                raw = info['raw']
                phys = info['phys']
                for i, (r_, p_) in enumerate(zip(raw, phys)):
                    kargs['text_{}'.format(i)] = 0
                    kargs['val_{}'.format(i)] = r_
                    conv_texts_tab['text_{}'.format(i)] = TextBlock(text=p_, meta=False)
                if info.get('default', b''):
                    conv_texts_tab['default_addr'] = TextBlock(text=info['default'], meta=False)
                kargs['default_addr'] = 0
                kargs['links_nr'] = len(raw) + 5
                gp_conv.append(ChannelConversion(**kargs))
            elif info and 'lower' in info:
                kargs = {}
                kargs['conversion_type'] = v4c.CONVERSION_TYPE_RTABX
                lower = info['lower']
                upper = info['upper']
                texts = info['phys']
                kargs['ref_param_nr'] = len(upper)
                kargs['default_addr'] = info.get('default', 0)
                kargs['links_nr'] = len(lower) + 5

                for i, (u_, l_, t_) in enumerate(zip(upper, lower, texts)):
                    kargs['lower_{}'.format(i)] = l_
                    kargs['upper_{}'.format(i)] = u_
                    kargs['text_{}'.format(i)] = 0
                    conv_texts_tab['text_{}'.format(i)] = TextBlock(text=t_, meta=False)
                if info.get('default', b''):
                    conv_texts_tab['default_addr'] = TextBlock(text=info['default'], meta=False)
                kargs['default_addr'] = 0
                gp_conv.append(ChannelConversion(**kargs))

            else:
                gp_conv.append(None)

            # source for channel
            gp_source.append(SourceInformation())

            # compute additional byte offset for large records size
            s_type, s_size = fmt_to_datatype(signal.samples.dtype)
            byte_size = max(s_size // 8, 1)
            min_val, max_val = get_min_max(signal.samples)
            kargs = {'channel_type': v4c.CHANNEL_TYPE_VALUE,
                     'bit_count': s_size,
                     'byte_offset': offset,
                     'bit_offset' : 0,
                     'data_type': s_type,
                     'min_raw_value': min_val if min_val<=max_val else 0,
                     'max_raw_value' : max_val if min_val<=max_val else 0,
                     'lower_limit' : min_val if min_val<=max_val else 0,
                     'upper_limit' : max_val if min_val<=max_val else 0}
            if min_val > max_val:
                kargs['flags'] = 0
            else:
                kargs['flags'] = v4c.FLAG_PHY_RANGE_OK | v4c.FLAG_VAL_RANGE_OK
            ch = Channel(**kargs)
            ch.name = name
            gp_channels.append(ch)
            gp['signal_data'].append(None)
            offset += byte_size

            if not name in self.channels_db:
                self.channels_db[name] = []
            self.channels_db[name].append((dg_cntr, ch_cntr))

            # update the parents as well
            field_name = get_unique_name(field_names, name)
            parents[ch_cntr] = field_name, 0

            fields.append(signal.samples)
            types.append( (field_name, signal.samples.dtype) )
            field_names.add(field_name)

            ch_cntr += 1

            # simple channels don't have channel dependencies
            gp_dep.append(None)

        canopen_time_fields = ('ms', 'days')
        canopen_date_fields = ('ms', 'min', 'hour', 'day', 'month', 'year', 'summer_time', 'day_of_week')
        for signal in composed_signals:
            names = signal.samples.dtype.names
            if names is None:
                names = []
            name = signal.name

            if names in (canopen_time_fields, canopen_date_fields):
                field_name = get_unique_name(field_names, name)
                field_names.add(field_name)

                if names == canopen_time_fields:

                    vals = signal.samples.tostring()

                    fields.append(frombuffer(vals, dtype='V6'))
                    types.append( (field_name, 'V6') )
                    byte_size = 6
                    s_type = v4c.DATA_TYPE_CANOPEN_TIME

                else:
                    vals = []
                    for field in ('ms', 'min', 'hour', 'day', 'month', 'year'):
                        vals.append(signal.samples[field])
                    vals = fromarrays(vals).tostring()

                    fields.append(frombuffer(vals, dtype='V7'))
                    types.append( (field_name, 'V7') )
                    byte_size = 7
                    s_type = v4c.DATA_TYPE_CANOPEN_DATE


                s_size = byte_size << 3

                # add channel texts
                for item in gp['texts'].values():
                    item.append({})
                gp_texts['channels'][-1]['name_addr'] = TextBlock(text=name, meta=False)
                if signal.unit:
                    gp_texts['channels'][-1]['unit_addr'] = TextBlock(text=signal.unit, meta=False)
                gp_texts['sources'][-1]['name_addr'] = si_text
                gp_texts['sources'][-1]['path_addr'] = si_text

                # add channel conversion
                gp_conv.append(None)

                # add channel source
                gp_source.append(SourceInformation())

                # there is no chanel dependency
                gp_dep.append(None)

                # add channel block
                kargs = {'channel_type': v4c.CHANNEL_TYPE_VALUE,
                         'bit_count': s_size,
                         'byte_offset': offset,
                         'bit_offset' : 0,
                         'data_type': s_type,
                         'min_raw_value': 0,
                         'max_raw_value' : 0,
                         'lower_limit' : 0,
                         'upper_limit' : 0,
                         'flags': 0}
                ch = Channel(**kargs)
                ch.name = name
                gp_channels.append(ch)
                gp['signal_data'].append(None)
                offset += byte_size

                if name in self.channels_db:
                    self.channels_db[name].append((dg_cntr, ch_cntr))
                else:
                    self.channels_db[name] = []
                    self.channels_db[name].append((dg_cntr, ch_cntr))

                # update the parents as well
                parents[ch_cntr] = field_name, 0

                ch_cntr += 1

            elif names and names[0] != signal.name:
                # here we have a structure channel composition

                field_name = get_unique_name(field_names, name)
                field_names.add(field_name)

                # first we add the structure channel
                # add channel texts
                for item in gp['texts'].values():
                    item.append({})
                gp_texts['channels'][-1]['name_addr'] = TextBlock(text=name, meta=False)
                if signal.unit:
                    gp_texts['channels'][-1]['unit_addr'] = TextBlock(text=signal.unit, meta=False)
                gp_texts['sources'][-1]['name_addr'] = si_text
                gp_texts['sources'][-1]['path_addr'] = si_text

                # add channel conversion
                gp_conv.append(None)

                # add channel source
                gp_source.append(SourceInformation())

                # add channel block
                kargs = {'channel_type': v4c.CHANNEL_TYPE_VALUE,
                         'bit_count': 8,
                         'byte_offset': offset,
                         'bit_offset' : 0,
                         'data_type': v4c.DATA_TYPE_BYTEARRAY,
                         'min_raw_value': 0,
                         'max_raw_value' : 0,
                         'lower_limit' : 0,
                         'upper_limit' : 0,
                         'flags': 0}
                ch = Channel(**kargs)
                ch.name = name
                gp_channels.append(ch)
                gp['signal_data'].append(None)

                if name not in self.channels_db:
                    self.channels_db[name] = []
                self.channels_db[name].append((dg_cntr, ch_cntr))

                # update the parents as well
                parents[ch_cntr] = name, 0

                ch_cntr += 1

                dep_list = []
                gp_dep.append(dep_list)

                # then we add the fields

                for name in names:
                    field_name = get_unique_name(field_names, name)
                    field_names.add(field_name)

                    samples = signal.samples[name]

                    s_type, s_size = fmt_to_datatype(samples.dtype)
                    byte_size = s_size >> 3

                    fields.append(samples)
                    types.append( (field_name, samples.dtype))
                    types.append(vals.dtype)

                    # add channel texts
                    for item in gp['texts'].values():
                        item.append({})
                    gp_texts['channels'][-1]['name_addr'] = TextBlock(text=name, meta=False)
                    gp_texts['sources'][-1]['name_addr'] = si_text
                    gp_texts['sources'][-1]['path_addr'] = si_text

                    # add channel conversion
                    gp_conv.append(None)

                    # add channel source
                    gp_source.append(SourceInformation())

                    min_val, max_val = get_min_max(samples)

                    # add channel block
                    kargs = {'channel_type': v4c.CHANNEL_TYPE_VALUE,
                             'bit_count': s_size,
                             'byte_offset': offset,
                             'bit_offset' : 0,
                             'data_type': s_type,
                             'min_raw_value': min_val,
                             'max_raw_value' : max_val,
                             'lower_limit' : min_val,
                             'upper_limit' : max_val,
                             'flags': v4c.FLAG_PHY_RANGE_OK | v4c.FLAG_VAL_RANGE_OK}
                    ch = Channel(**kargs)

                    dep_list.append(ch)
                    ch.name = name
                    gp_channels.append(ch)
                    gp['signal_data'].append(None)
                    offset += byte_size

                    if name not in self.channels_db:
                        self.channels_db[name] = []
                    self.channels_db[name].append((dg_cntr, ch_cntr))

                    # update the parents as well
                    parents[ch_cntr] = field_name, 0

                    ch_cntr += 1

            else:
                # here we have channel arrays or mdf version 3 channel dependencies
                if names:
                    samples = signal.samples[names[0]]
                else:
                    samples = signal.samples
                shape = samples.shape[1:]

                if len(shape) > 1:
                    # add channel dependency block for composed parent channel
                    dims_nr = len(shape)
                    names_nr = len(names)

                    if names_nr == 0:
                        kargs = {'dims': dims_nr,
                                 'ca_type': v4c.CA_TYPE_LOOKUP,
                                 'flags': v4c.FLAG_CA_FIXED_AXIS,
                                 'byte_offset_base': samples.dtype.itemsize,
                                 }
                        for i in range(dims_nr):
                            kargs['dim_size_{}'.format(i)] = shape[i]

                    elif len(names) == 1:
                        kargs = {'dims': dims_nr,
                                 'ca_type': v4c.CA_TYPE_ARRAY,
                                 'flags': 0,
                                 'byte_offset_base': samples.dtype.itemsize,
                                 }
                        for i in range(dims_nr):
                            kargs['dim_size_{}'.format(i)] = shape[i]

                    else:
                        kargs = {'dims': dims_nr,
                                 'ca_type': v4c.CA_TYPE_LOOKUP,
                                 'flags': v4c.FLAG_CA_AXIS,
                                 'byte_offset_base': samples.dtype.itemsize,
                                 }
                        for i in range(dims_nr):
                            kargs['dim_size_{}'.format(i)] = shape[i]

                    parent_dep = ChannelArrayBlock(**kargs)
                    gp_dep.append([parent_dep,])

                else:
                    # add channel dependency block for composed parent channel
                    kargs = {'dims': 1,
                             'ca_type': v4c.CA_TYPE_SCALE_AXIS,
                             'flags': 0,
                             'byte_offset_base': samples.dtype.itemsize,
                             'dim_size_0': shape[0]}
                    parent_dep = ChannelArrayBlock(**kargs)
                    gp_dep.append([parent_dep,])

                field_name = get_unique_name(field_names, name)
                field_names.add(field_name)

                fields.append(samples)
                types.append( (field_name, samples.dtype, shape) )


                # first we add the structure channel
                # add channel texts
                for item in gp['texts'].values():
                    item.append({})
                gp_texts['channels'][-1]['name_addr'] = TextBlock(text=name, meta=False)
                if signal.unit:
                    gp_texts['channels'][-1]['unit_addr'] = TextBlock(text=signal.unit, meta=False)
                gp_texts['sources'][-1]['name_addr'] = si_text
                gp_texts['sources'][-1]['path_addr'] = si_text

                # add channel conversion
                gp_conv.append(None)

                # add channel source
                gp_source.append(SourceInformation())


                s_type, s_size = fmt_to_datatype(samples.dtype)

                # add channel block
                kargs = {'channel_type': v4c.CHANNEL_TYPE_VALUE,
                         'bit_count': s_size,
                         'byte_offset': offset,
                         'bit_offset' : 0,
                         'data_type': s_type,
                         'min_raw_value': 0,
                         'max_raw_value' : 0,
                         'lower_limit' : 0,
                         'upper_limit' : 0,
                         'flags': 0}
                ch = Channel(**kargs)
                ch.name = name
                gp_channels.append(ch)
                gp['signal_data'].append(None)

                size = s_size >> 3
                for dim in shape:
                    size *= dim
                offset += size

                if name not in self.channels_db:
                    self.channels_db[name] = []
                self.channels_db[name].append((dg_cntr, ch_cntr))

                # update the parents as well
                parents[ch_cntr] = name, 0

                ch_cntr += 1

                for name in names[1:]:
                    field_name = get_unique_name(field_names, name)
                    field_names.add(field_name)

                    samples = signal.samples[name]
                    shape = samples.shape[1:]
                    fields.append(samples)
                    types.append( (field_name, samples.dtype, shape) )

                    # add composed parent signal texts
                    for item in gp['texts'].values():
                        item.append({})
                    gp_texts['channels'][-1]['name_addr'] = TextBlock(text=name, meta=False)
                    gp_texts['sources'][-1]['name_addr'] = si_text
                    gp_texts['sources'][-1]['path_addr'] = si_text

                    # add channel conversion and source
                    gp_conv.append(None)
                    gp_source.append(SourceInformation())
                    # add channel dependency block
                    kargs = {'dims': 1,
                             'ca_type': v4c.CA_TYPE_SCALE_AXIS,
                             'flags': 0,
                             'byte_offset_base': samples.dtype.itemsize,
                             'dim_size_0': shape[0]}
                    gp_dep.append([ChannelArrayBlock(**kargs),])

                    # add components channel
                    min_val, max_val = get_min_max(samples)
                    s_type, s_size = fmt_to_datatype(samples.dtype)
                    byte_size = max(s_size // 8, 1)
                    kargs = {'channel_type': v4c.CHANNEL_TYPE_VALUE,
                             'bit_count': s_size,
                             'byte_offset': offset,
                             'bit_offset' : 0,
                             'data_type': s_type,
                             'min_raw_value': min_val if min_val<=max_val else 0,
                             'max_raw_value' : max_val if min_val<=max_val else 0,
                             'lower_limit' : min_val if min_val<=max_val else 0,
                             'upper_limit' : max_val if min_val<=max_val else 0,
                             'flags': v4c.FLAG_PHY_RANGE_OK | v4c.FLAG_VAL_RANGE_OK
                             }

                    channel = Channel(**kargs)
                    channel.name = name
                    gp_channels.append(channel)
                    parent_dep.referenced_channels.append((ch_cntr, dg_cntr))
                    gp['signal_data'].append(None)
                    for dim in shape:
                        byte_size *= dim
                    offset += byte_size

                    if name not in self.channels_db:
                        self.channels_db[name] = []
                    self.channels_db[name].append((dg_cntr, ch_cntr))

                    # update the parents as well
                    parents[ch_cntr] = field_name, 0

                    ch_cntr += 1

        #channel group
        kargs = {'cycles_nr': cycles_nr,
                 'samples_byte_nr': offset}
        gp['channel_group'] = ChannelGroup(**kargs)
        gp['size'] = cycles_nr * offset

        #data group
        gp['data_group'] = DataGroup()

        #data block
        if PYVERSION == 2:
            types = fix_dtype_fields(types)
        types = dtype(types)


        gp['sorted'] = True
        gp['types'] = types
        gp['parents'] = parents

        samples = fromarrays(fields, dtype=types)
        block = samples.tostring()


        if self.load_measured_data:
            gp['data_location'] = v4c.LOCATION_MEMORY
            gp['data_block'] = DataBlock(data=block)
        else:
            gp['data_location'] = v4c.LOCATION_TEMPORARY_FILE
            if self._tempfile is None:
                self._tempfile = TemporaryFile()
            self._tempfile.seek(0, v4c.SEEK_END)
            data_address = self._tempfile.tell()
            gp['data_group']['data_block_addr'] = data_address
            self._tempfile.write(bytes(block))

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
        fh_text = TextBlock(text="""<FHcomment>
	<TX>Added new embedded attachment from {}</TX>
	<tool_id>asammdf</tool_id>
	<tool_vendor>asammdf</tool_vendor>
	<tool_version>2.0.0</tool_version>
</FHcomment>""".format(file_name if file_name else 'bin.bin'), meta=True)

        self.file_history.append((fh, fh_text))

        texts = {}
        texts['mime_addr'] = TextBlock(text=mime, meta=False)
        if comment:
            texts['comment_addr'] = TextBlock(text=comment, meta=False)
        texts['file_name_addr'] = TextBlock(text=file_name if file_name else 'bin.bin')
        at_block = AttachmentBlock(data=data, compression=compression)
        at_block['creator_index'] = creator_index
        self.attachments.append((at_block, texts))

    def close(self):
        """ if the MDF was created with load_measured_data=False and new channels
        have been appended, then this must be called just before the object is not
        used anymore to clean-up the temporary file"""
        if self.load_measured_data == False and self._tempfile is not None:
            self._tempfile.close()

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
            if flags & v4c.FLAG_AT_EMBEDDED:
                data = attachment.extract()

                file_path = texts['file_name_addr']['text'].decode('utf-8').strip(' \n\t\x00')
                out_path = os.path.dirname(file_path)
                if out_path:
                    if not os.path.exists(out_path):
                        os.makedirs(out_path)

                with open(file_path, 'wb') as f:
                    f.write(data)

                return data
            else:
                # for external attachemnts read the files and return the content
                if flags & v4c.FLAG_AT_MD5_VALID:
                    file_path = texts['file_name_addr']['text'].decode('utf-8').strip(' \n\t\x00')
                    data = open(file_path, 'rb').read()
                    md5_worker = md5()
                    md5_worker.update(data)
                    md5_sum = md5_worker.digest()
                    if attachment['md5_sum'] == md5_sum:
                        if texts['mime_addr']['text'].decode('utf-8').startswith('text'):
                            with open(file_path, 'r') as f:
                                data = f.read()
                        return data
                    else:
                        warnings.warn('ATBLOCK md5sum="{}" and external attachment data ({}) md5sum="{}"'.format(attachment['md5_sum'], file_path, md5_sum))
                else:
                    if texts['mime_addr']['text'].decode('utf-8').startswith('text'):
                        mode = 'r'
                    else:
                        mode = 'rb'
                    with open(file_path, mode) as f:
                        data = f.read()
                    return data
        except Exception as err:
            os.chdir(current_path)
            warnings.warn('Exception during attachment extraction: ' + repr(err))

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
            returns *Signal* if *samples_only*=*False* (default option), otherwise returns numpy.array
            The *Signal* samples are:

                * numpy recarray for channels that have composition/channel array address or for channel of type BYTEARRAY, CANOPENDATE, CANOPENTIME
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
                        if (gp_nr, ch_nr) == (group, index):
                            break
                    else:
                        gp_nr, ch_nr = self.channels_db[name][0]
                        warnings.warn('You have selected group "{}" for channel "{}", but this channel was not found in this group. Using first occurance of "{}" from group "{}"'.format(group, name, name, gp_nr))

        grp = self.groups[gp_nr]
        channel = grp['channels'][ch_nr]
        conversion = grp['channel_conversions'][ch_nr]
        dependency_list = grp['channel_dependencies'][ch_nr]

        # get data group record
        try:
            parents, dtypes = grp['parents'], grp['types']
        except KeyError:
            grp['parents'], grp['types'] = self._prepare_record(grp)
            parents, dtypes = grp['parents'], grp['types']

        # get group data
        data = self._load_group_data(grp)

        info = None

        # get the channel signal data if available
        if self.load_measured_data:
            signal_data = grp['signal_data'][ch_nr]
            if signal_data:
                signal_data = signal_data['data']
            else:
                signal_data = b''
        else:
            with open(self.name, 'rb') as original_file:
                signal_data = self._read_agregated_signal_data(channel['data_block_addr'], original_file)

        info = None

        # check if this is a channel array
        if dependency_list:
            arrays = []
            name = channel.name

            if all(isinstance(dep, Channel) for dep in dependency_list):
                # structure channel composition
                arrays = [self.get(ch.name, samples_only=True) for ch in dependency_list]
                names = [ch.name for ch in dependency_list]
                types = [ (name, arr.dtype) for name, arr in zip(names, arrays)]
                if PYVERSION == 2:
                    types = fix_dtype_fields(types)
                types = dtype(types)

                vals = fromarrays(arrays, dtype=types)

                cycles_nr = len(vals)

            else:
                # channel arrays
                arrays = []
                types = []

                try:
                    parent, bit_offset = parents[ch_nr]
                except:
                    parent, bit_offset = None, None

                if parent is not None:
                    if 'record' not in grp:
                        if dtypes.itemsize:
                            record = fromstring(data, dtype=dtypes)
                        else:
                            record = None

                        if self.load_measured_data:
                            grp['record'] = record
                    else:
                        record = grp['record']

                    vals = record[parent]
                else:
                    vals = self._get_not_byte_aligned_data(data, grp, ch_nr)

                dep = dependency_list[0]
                if dep['flags'] & v4c.FLAG_CA_INVERSE_LAYOUT:

                    shape = vals.shape
                    shape = (shape[0],) + shape[1:][::-1]
                    vals = vals.reshape(shape)

                    axes = (0,) + tuple(range(len(shape) - 1, 0, -1))
                    vals = transpose(vals, axes=axes)

                cycles_nr = len(vals)

                for ca_block in dependency_list[:1]:
                    dims_nr = ca_block['dims']

                    if ca_block['ca_type'] == v4c.CA_TYPE_SCALE_AXIS:
                        shape = (ca_block['dim_size_0'], )
                        arrays.append(vals)
                        types.append( (channel.name, vals.dtype, shape) )

                    elif ca_block['ca_type'] == v4c.CA_TYPE_LOOKUP:
                        shape = vals.shape[1:]
                        arrays.append(vals)
                        types.append( (channel.name, vals.dtype, shape) )

                        if ca_block['flags'] & v4c.FLAG_CA_FIXED_AXIS:
                            for i in range(dims_nr):
                                shape = (ca_block['dim_size_{}'.format(i)], )
                                axis = []
                                for j in range(shape[0]):
                                    axis.append(ca_block['axis_{}_value_{}'.format(i, j)])
                                axis = array([axis for _ in range(cycles_nr)])
                                arrays.append(axis)
                                types.append( ('axis_{}'.format(i), axis.dtype, shape) )
                        else:
                            for i in range(dims_nr):
                                ch_nr, dg_nr = ca_block.referenced_channels[i]
                                axis = self.groups[dg_nr]['channels'][ch_nr]
                                shape = (ca_block['dim_size_{}'.format(i)], )
                                axis_values = self.get(group=dg_nr, index=ch_nr, samples_only=True)[axis.name]
                                arrays.append(axis_values)
                                types.append( (axis.name, axis_values.dtype, shape))

                    elif ca_block['ca_type'] == v4c.CA_TYPE_ARRAY:
                        shape = vals.shape[1:]
                        arrays.append(vals)
                        types.append( (channel.name, vals.dtype, shape) )

                for ca_block in dependency_list[1:]:
                    dims_nr = ca_block['dims']

                    if ca_block['flags'] & v4c.FLAG_CA_FIXED_AXIS:
                        for i in range(dims_nr):
                            shape = (ca_block['dim_size_{}'.format(i)], )
                            axis = []
                            for j in range(shape[0]):
                                axis.append(ca_block['axis_{}_value_{}'.format(i, j)])
                            axis = array([axis for _ in range(cycles_nr)])
                            arrays.append(axis)
                            types.append( ('axis_{}'.format(i), axis.dtype, shape) )
                    else:
                        for i in range(dims_nr):
                            ch_nr, dg_nr = ca_block.referenced_channels[i]
                            axis = self.groups[dg_nr]['channels'][ch_nr]
                            shape = (ca_block['dim_size_{}'.format(i)], )
                            axis_values = self.get(group=dg_nr, index=ch_nr, samples_only=True)[axis.name]
                            arrays.append(axis_values)
                            types.append( (axis.name, axis_values.dtype, shape))


                if PYVERSION == 2:
                    types = fix_dtype_fields(types)

                vals = fromarrays(arrays, dtype(types))
        else:
            # get channel values
            if channel['channel_type'] in (v4c.CHANNEL_TYPE_VIRTUAL, v4c.CHANNEL_TYPE_VIRTUAL_MASTER):
                data_type = channel['data_type']
                ch_dtype = dtype(get_fmt(data_type, 8))
                cycles = grp['channel_group']['cycles_nr']
                vals = arange(cycles, dtype=ch_dtype)
            else:
                try:
                    parent, bit_offset = parents[ch_nr]
                except:
                    parent, bit_offset = None, None

                if parent is not None:
                    if 'record' not in grp:
                        if dtypes.itemsize:
                            record = fromstring(data, dtype=dtypes)
                        else:
                            record = None

                        if self.load_measured_data:
                            grp['record'] = record
                    else:
                        record = grp['record']

                    vals = record[parent]
                    bits = channel['bit_count']
                    size = vals.dtype.itemsize
                    data_type = channel['data_type']

                    if bit_offset:
                        dtype_= vals.dtype
                        if issubdtype(dtype_, signedinteger):
                            vals = vals.astype(dtype('<u{}'.format(size)))
                            vals >>= bit_offset
                        else:
                            vals = vals >> bit_offset

                    if not bits == size * 8:
                        mask = (1<<bits) - 1
                        if vals.flags.writeable:
                            vals &= mask
                        else:
                            vals = vals & mask
                        if data_type in v4c.SIGNED_INT:
                            size = vals.dtype.itemsize
                            mask = (1 << (size * 8)) - 1
                            mask = (mask << bits) & mask
                            vals |= mask
                            vals = vals.astype('<i{}'.format(size), copy=False)
                else:
                    vals = self._get_not_byte_aligned_data(data, grp, ch_nr)

            conversion_type = v4c.CONVERSION_TYPE_NON if conversion is None else conversion['conversion_type']

            if conversion_type == v4c.CONVERSION_TYPE_NON:
                # check if it is VLDS channel type with SDBLOCK

                if channel['channel_type'] in (v4c.CHANNEL_TYPE_VALUE, v4c.CHANNEL_TYPE_MLSD):
                    if v4c.DATA_TYPE_STRING_LATIN_1 <= channel['data_type'] <= v4c.DATA_TYPE_STRING_UTF_16_BE:
                        vals = [val.tobytes() for val in vals]

                        if channel['data_type'] == v4c.DATA_TYPE_STRING_UTF_16_BE:
                            encoding = 'utf-16-be'

                        elif channel['data_type'] == v4c.DATA_TYPE_STRING_UTF_16_LE:
                            encoding = 'utf-16-le'

                        elif channel['data_type'] == v4c.DATA_TYPE_STRING_UTF_8:
                            encoding = 'utf-8'

                        elif channel['data_type'] == v4c.DATA_TYPE_STRING_LATIN_1:
                            encoding = 'latin-1'

                        vals = array([x.decode(encoding).strip('\x00') for x in vals])
                        vals = encode(vals, 'latin-1')

                    # CANopen date
                    elif channel['data_type'] == v4c.DATA_TYPE_CANOPEN_DATE:

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
                    elif channel['data_type'] == v4c.DATA_TYPE_CANOPEN_TIME:
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
                    elif channel['data_type'] == v4c.DATA_TYPE_BYTEARRAY:
                        vals = vals.tostring()
                        size = max(bits>>3, 1)

                        vals = frombuffer(vals, dtype=dtype('({},)u1'.format(size)))

                        types = [(channel.name, vals.dtype, vals.shape[1:])]
                        if PYVERSION == 2:
                            types = fix_dtype_fields(types)

                        types = dtype(types)
                        arrays = [vals, ]

                        vals = fromarrays(arrays, dtype=types)

                elif channel['channel_type'] == v4c.CHANNEL_TYPE_VLSD:
                    if signal_data:
                        values = []
                        for offset in vals:
                            offset = int(offset)
                            str_size = unpack_from('<I', signal_data, offset)[0]
                            values.append(signal_data[offset+4: offset+4+str_size])

                        if channel['data_type'] == v4c.DATA_TYPE_STRING_UTF_16_BE:
                            vals = [v.decode('utf-16-be') for v in values]

                        elif channel['data_type'] == v4c.DATA_TYPE_STRING_UTF_16_LE:
                            vals = [v.decode('utf-16-le') for v in values]

                        elif channel['data_type'] == v4c.DATA_TYPE_STRING_UTF_8:
                            vals = [v.decode('utf-8') for v in values]

                        elif channel['data_type'] == v4c.DATA_TYPE_STRING_LATIN_1:
                            vals = [v.decode('latin-1') for v in values]

                        if PYVERSION == 2:
                            vals = array([str(val) for val in vals])
                        else:
                            vals = array(vals)

                        vals = encode(vals, 'latin-1')
                    else:
                        # no VLSD signal data samples
                        vals = array([])

            elif conversion_type == v4c.CONVERSION_TYPE_LIN:
                a = conversion['a']
                b = conversion['b']
                if (a, b) != (1, 0):
                    vals = vals * a
                    if b:
                        vals += b

            elif conversion_type == v4c.CONVERSION_TYPE_RAT:
                P1 = conversion['P1']
                P2 = conversion['P2']
                P3 = conversion['P3']
                P4 = conversion['P4']
                P5 = conversion['P5']
                P6 = conversion['P6']
                if not (P1, P2, P3, P4, P5, P6) == (0, 1, 0, 0, 0, 1):
                    X = vals
                    vals = evaluate('(P1 * X**2 + P2 * X + P3) / (P4 * X**2 + P5 * X + P6)')

            elif conversion_type == v4c.CONVERSION_TYPE_ALG:
                formula = grp['texts']['conversions'][ch_nr]['formula_addr']['text'].decode('utf-8').strip(' \n\t\x00')
                X = vals
                vals = evaluate(formula)

            elif conversion_type in (v4c.CONVERSION_TYPE_TABI, v4c.CONVERSION_TYPE_TAB):
                nr = conversion['val_param_nr'] // 2
                raw = array([conversion['raw_{}'.format(i)] for i in range(nr)])
                phys = array([conversion['phys_{}'.format(i)] for i in range(nr)])
                if conversion_type == v4c.CONVERSION_TYPE_TABI:
                    vals = interp(vals, raw, phys)
                else:
                    idx = searchsorted(raw, vals)
                    idx = clip(idx, 0, len(raw) - 1)
                    vals = phys[idx]

            elif conversion_type == v4c.CONVERSION_TYPE_RTAB:
                nr = (conversion['val_param_nr'] - 1) // 3
                lower = array([conversion['lower_{}'.format(i)] for i in range(nr)])
                upper = array([conversion['upper_{}'.format(i)] for i in range(nr)])
                phys = array([conversion['phys_{}'.format(i)] for i in range(nr)])
                default = conversion['default']

                # INT channel
                if channel['data_type'] <= 3:

                    res = []
                    for v in vals:
                        for l, u, p in zip(lower, upper, phys):
                            if l <= v <= u:
                                res.append(p)
                                break
                        else:
                            res.append(default)
                    size = max(bits>>3, 1)
                    ch_fmt = get_fmt(channel['data_type'], size)
                    vals = array(res).astype(ch_fmt)

                # else FLOAT channel
                else:
                    res = []
                    for v in vals:
                        for l, u, p in zip(lower, upper, phys):
                            if l <= v < u:
                                res.append(p)
                                break
                        else:
                            res.append(default)
                    size = max(bits>>3, 1)
                    ch_fmt = get_fmt(channel['data_type'], size)
                    vals = array(res).astype(ch_fmt)

            elif conversion_type == v4c.CONVERSION_TYPE_TABX:
                nr = conversion['val_param_nr']
                raw = array([conversion['val_{}'.format(i)] for i in range(nr)])
                phys = array([grp['texts']['conversion_tab'][ch_nr]['text_{}'.format(i)]['text'] for i in range(nr)])
                default = grp['texts']['conversion_tab'][ch_nr].get('default_addr', {}).get('text', b'')
                info = {'raw': raw, 'phys': phys, 'default': default}

            elif conversion_type == v4c.CONVERSION_TYPE_RTABX:
                nr = conversion['val_param_nr'] // 2

                phys = array([grp['texts']['conversion_tab'][ch_nr]['text_{}'.format(i)]['text'] for i in range(nr)])
                lower = array([conversion['lower_{}'.format(i)] for i in range(nr)])
                upper = array([conversion['upper_{}'.format(i)] for i in range(nr)])
                default = grp['texts']['conversion_tab'][ch_nr].get('default_addr', {}).get('text', b'')
                info = {'lower': lower, 'upper': upper, 'phys': phys, 'default': default}

            elif conversion == v4c.CONVERSION_TYPE_TTAB:
                nr = conversion['val_param_nr'] - 1

                raw = array([grp['texts']['conversion_tab'][ch_nr]['text_{}'.format(i)]['text'] for i in range(nr)])
                phys = array([conversion['val_{}'.format(i)] for i in range(nr)])
                default = conversion['val_default']
                info = {'lower': lower, 'upper': upper, 'phys': phys, 'default': default}

            elif conversion == v4c.CONVERSION_TYPE_TRANS:
                nr = (conversion['ref_param_nr'] - 1 ) // 2
                in_ = array([grp['texts']['conversion_tab'][ch_nr]['input_{}'.format(i)]['text'] for i in range(nr)])
                out_ = array([grp['texts']['conversion_tab'][ch_nr]['output_{}'.format(i)]['text'] for i in range(nr)])
                default = grp['texts']['conversion_tab'][ch_nr]['default_addr']['text']

                res = []
                for v in vals:
                    for i, o in zip(in_, out_):
                        if v == i:
                            res.append(o)
                            break
                    else:
                        res.append(default)
                vals = array(res)
                info = {'input': in_, 'output': out_, 'default': default}

        # in case of invalidation bits, valid_index will hold the valid indexes
        valid_index = None
        if grp['channel_group']['invalidation_bytes_nr']:

            if channel['flags'] & ( v4c.FLAG_INVALIDATION_BIT_VALID | v4c.FLAG_ALL_SAMPLES_VALID ) == v4c.FLAG_INVALIDATION_BIT_VALID:

                ch_invalidation_pos = channel['pos_invalidation_bit']
                pos_byte, pos_offset = divmod(ch_invalidation_pos)
                mask = 1 << pos_offset

                inval_bytes = record['invalidation_bytes']
                inval_index = array([bytes_[pos_byte] & mask for bytes_ in inval_bytes])
                valid_index = argwhere(inval_index == 0).flatten()
                vals = vals[valid_index]

        if samples_only:
            res = vals
        else:
            # search for unit in conversion texts
            conv_texts = grp['texts']['conversions'][ch_nr]
            channel_texts = grp['texts']['channels'][ch_nr]

            if 'unit_addr' in conv_texts:
                unit = conv_texts['unit_addr']
                if PYVERSION == 3:
                    try:
                        unit = unit['text'].decode('utf-8').strip(' \n\t\x00')
                    except:
                        unit = ''
                else:
                    unit = unit['text'].strip(' \n\t\x00')
            else:
                # search for physical unit in channel texts
                if 'unit_addr' in channel_texts:
                    unit = channel_texts['unit_addr']
                    if PYVERSION == 3:
                        unit = unit['text'].decode('utf-8').strip(' \n\t\x00')
                    else:
                        unit = unit['text'].strip(' \n\t\x00')
                else:
                    unit = ''


            # get the channel commment if available
            if 'comment_addr' in channel_texts:
                comment = channel_texts['comment_addr']
                if comment['id'] == b'##MD':
                    comment = comment['text'].decode('utf-8').strip(' \n\t\x00')
                    try:
                        comment = XML.fromstring(comment).find('TX').text
                    except:
                        comment = ''
                else:
                    comment = comment['text'].decode('utf-8')
            else:
                comment = ''

            t = self.get_master(gp_nr, data)

            # consider invalidation bits
            if valid_index is not None:
                t = t[valid_index]

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

    def get_master(self, index, data=None):
        """ returns master channel samples for given group

        Parameters
        ----------
        index : int
            group index
        data : bytes
            data block raw bytes; default None

        Returns
        -------
        t : numpy.array
            master channel samples

        """
        try:
            return self._master_channel_cache[index]
        except:
            pass

        group = self.groups[index]

        try:
            time_ch_nr = self.masters_db[index]
        except:
            time_ch_nr = None
        cycles_nr = group['channel_group']['cycles_nr']

        if time_ch_nr is None:
            t = arange(cycles_nr, dtype=float64)
        else:
            time_conv = group['channel_conversions'][time_ch_nr]
            time_ch = group['channels'][time_ch_nr]
            if time_ch['channel_type'] == v4c.CHANNEL_TYPE_VIRTUAL_MASTER:
                time_a = time_conv['a']
                time_b = time_conv['b']
                t = arange(cycles_nr, dtype=float64) * time_a + time_b
            else:
                # get data group parents and dtypes
                try:
                    parents, dtypes = group['parents'], group['types']
                except KeyError:
                    group['parents'], group['types'] = self._prepare_record(group)
                    parents, dtypes = group['parents'], group['types']

                # get data
                if data is None:
                    data = self._load_group_data(group)

                try:
                    parent, bit_offset = parents[time_ch_nr]
                except:
                    parent, bit_offset = None, None
                if parent is not None:
                    not_found = object()
                    record = group.get('record', not_found)
                    if record is not_found:
                        if dtypes.itemsize:
                            record = fromstring(data, dtype=dtypes)
                        else:
                            record = None

                        if self.load_measured_data:
                            group['record'] = record
                    t = record[parent]
                else:
                    t = self._get_not_byte_aligned_data(data, group, time_ch_nr)

                # get timestamps
                if time_conv and time_conv['conversion_type'] == v4c.CONVERSION_TYPE_LIN:
                    time_a = time_conv['a']
                    time_b = time_conv['b']
                    t = t * time_a
                    if time_b:
                        t += time_b
        self._master_channel_cache[index] = t
        return t

    def info(self):
        """get MDF information as a dict

        Examples
        --------
        >>> mdf = MDF4('test.mdf')
        >>> mdf.info()


        """
        info = {}
        info['version'] = self.identification['version_str'].strip(b'\x00').decode('utf-8').strip(' \n\t\x00')
        info['groups'] = len(self.groups)
        for i, gp in enumerate(self.groups):
            inf = {}
            info['group {}'.format(i)] = inf
            inf['cycles'] = gp['channel_group']['cycles_nr']
            inf['channels count'] = len(gp['channels'])
            for j, ch in enumerate(gp['channels']):
                inf['channel {}'.format(j)] = (ch.name, ch['channel_type'])

        return info

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
            use compressed data blocks, default 0; only valid since version 4.10

            * 0 - no compression
            * 1 - deflate (slower, but produces smaller files)
            * 2 - transposition + deflate (slowest, but produces the smallest files)

        """
        if self.name is None and dst == '':
            raise MdfException('Must specify a destination file name for MDF created from scratch')

        dst = dst if dst else self.name
        if overwrite == False:
            if os.path.isfile(dst):
                cntr = 0
                while True:
                    name = os.path.splitext(dst)[0] + '_{}.mf4'.format(cntr)
                    if not os.path.isfile(name):
                        break
                    else:
                        cntr += 1
                warnings.warn('Destination file "{}" already exists and "overwrite" is False. Saving MDF file as "{}"'.format(dst, name))
                dst = name

        if not self.file_history:
            comment = 'created'
        else:
            comment = 'updated'

        self.file_history.append([FileHistory(), TextBlock(text='<FHcomment>\n<TX>{}</TX>\n<tool_id>PythonMDFEditor</tool_id>\n<tool_vendor></tool_vendor>\n<tool_version>1.0</tool_version>\n</FHcomment>'.format(comment), meta=True)])
        with open(dst, 'wb') as dst:
            defined_texts = {}

            write = dst.write
            tell = dst.tell
            seek = dst.seek

            write(bytes(self.identification))
            write(bytes(self.header))

            original_data_addresses = []

            # write DataBlocks first
            for gp in self.groups:

                original_data_addresses.append(gp['data_group']['data_block_addr'])
                address = tell()

                if self.load_measured_data:
                    data_block = gp['data_block']
                    if compression and self.version != '4.00':
                        kargs = {'data': data_block['data'],
                                 'zip_type': v4c.FLAG_DZ_DEFLATE if compression == 1 else v4c.FLAG_DZ_TRANPOSED_DEFLATE,
                                 'param': 0 if compression == 1 else gp['channel_group']['samples_byte_nr']}
                        data_block = DataZippedBlock(**kargs)
                    write(bytes(data_block))

                    align = data_block['block_len'] % 8
                    if align:
                        write(b'\x00' * (8-align))
                else:
                    # trying to call bytes([gp, address]) will result in an exception
                    # that be used as a flag for non existing data block in case
                    # of load_measured_data=False, the address is the actual address
                    # of the data group's data within the original file
                    # this will only be executed for data blocks when load_measured_data=False

                    data = self._load_group_data(gp)
                    if compression and self.version != '4.00':
                        kargs = {'data': data,
                                 'zip_type': v4c.FLAG_DZ_DEFLATE if compression == 1 else v4c.FLAG_DZ_TRANPOSED_DEFLATE,
                                 'param': 0 if compression == 1 else gp['channel_group']['samples_byte_nr']}
                        data_block = DataZippedBlock(**kargs)
                    else:
                        data_block = DataBlock(data=data)
                    write(bytes(data_block))

                    align = data_block['block_len'] % 8
                    if align:
                        write(b'\x00' * (8-align))

                gp['data_group']['data_block_addr'] = address if gp['channel_group']['cycles_nr'] else 0

            address = tell()

            blocks = []

            if self.file_comment:
                self.file_comment.address = address
                address += self.file_comment['block_len']
                blocks.append(self.file_comment)

            # attachemnts
            if self.attachments:
                for at_block, texts in self.attachments:
                    for key, text in texts.items():
                        at_block[key] = text.address = address
                        address += text['block_len']
                        blocks.append(text)

                for at_block, texts in self.attachments:
                    at_block.address = address
                    blocks.append(at_block)
                    align = at_block['block_len'] % 8
                    # append 8vyte alignemnt bytes for attachments
                    if align % 8:
                        blocks.append(b'\x00' * (8 - align))
                    address += at_block['block_len'] + align

                for i, (at_block, text) in enumerate(self.attachments[:-1]):
                    at_block['next_at_addr'] = self.attachments[i+1][0].address
                self.attachments[-1][0]['next_at_addr'] = 0

            # file history blocks
            for i, (fh, fh_text) in enumerate(self.file_history):
                fh_text.address = address
                blocks.append(fh_text)
                address += fh_text['block_len']

                fh['comment_addr'] = fh_text.address

            for i, (fh, fh_text) in enumerate(self.file_history):
                fh.address = address
                address += fh['block_len']
                blocks.append(fh)

            for i, (fh, fh_text) in enumerate(self.file_history[:-1]):
                fh['next_fh_addr'] = self.file_history[i+1][0].address
            self.file_history[-1][0]['next_fh_addr'] = 0

            # data groups
            for gp in self.groups:
                gp['data_group'].address = address
                address += gp['data_group']['block_len']
                blocks.append(gp['data_group'])

                gp['data_group']['comment_addr'] = 0

            for i, dg in enumerate(self.groups[:-1]):
                dg['data_group']['next_dg_addr'] = self.groups[i+1]['data_group'].address
            self.groups[-1]['data_group']['next_dg_addr'] = 0

            # go through each data group and append the rest of the blocks
            for i, gp in enumerate(self.groups):
                # write TXBLOCK's
                for item_list in gp['texts'].values():
                    for dict_ in item_list:
                        for key, tx_block in dict_.items():
                            #text blocks can be shared
                            text = tx_block['text']
                            if text in defined_texts:
                                tx_block.address = defined_texts[text]
                            else:
                                defined_texts[text] = address
                                tx_block.address = address
                                address += tx_block['block_len']
                                blocks.append(tx_block)

                # channel conversions
                for j, conv in enumerate(gp['channel_conversions']):
                    if conv:
                        conv.address = address
                        conv_texts = gp['texts']['conversions'][j]

                        for key, text_block in conv_texts.items():
                            conv[key] = text_block.address
                        conv['inv_conv_addr'] = 0

                        if conv['conversion_type'] in (v4c.CONVERSION_TYPE_TABX,
                                                       v4c.CONVERSION_TYPE_RTABX,
                                                       v4c.CONVERSION_TYPE_TTAB,
                                                       v4c.CONVERSION_TYPE_TRANS):
                            for key in gp['texts']['conversion_tab'][j]:
                                conv[key] = gp['texts']['conversion_tab'][j][key].address

                        address += conv['block_len']
                        blocks.append(conv)

                # channel sources
                for j, source in enumerate(gp['channel_sources']):
                    if source:
                        source.address = address
                        source_texts = gp['texts']['sources'][j]

                        for key in ('name_addr', 'path_addr', 'comment_addr'):
                            try:
                                source[key] = source_texts[key].address
                            except:
                                source[key] = 0

                        address += source['block_len']
                        blocks.append(source)

                # channel data
                for j, signal_data in enumerate(gp['signal_data']):
                    if signal_data:
                        signal_data.address = address
                        address += signal_data['block_len']
                        blocks.append(signal_data)

                # channel dependecies
                for j, dep_list in enumerate(gp['channel_dependencies']):
                    if dep_list:
                        if all(isinstance(dep, ChannelArrayBlock) for dep in dep_list):
                            for dep in dep_list:
                                dep.address = address
                                address += dep['block_len']
                                blocks.append(dep)

                # channels
                for j, (channel, signal_data) in enumerate(zip(gp['channels'], gp['signal_data'])):
                    channel.address = address
                    channel_texts = gp['texts']['channels'][j]

                    address += channel['block_len']
                    blocks.append(channel)

                    for key in ('comment_addr', 'unit_addr'):
                        if key in channel_texts:
                            channel[key] = channel_texts[key].address
                        else:
                            channel[key] = 0
                    channel['name_addr'] = channel_texts['name_addr'].address

                    channel['conversion_addr'] = 0 if not gp['channel_conversions'][j] else gp['channel_conversions'][j].address
                    channel['source_addr'] = gp['channel_sources'][j].address if gp['channel_sources'][j] else 0
                    channel['data_block_addr'] = signal_data.address if signal_data else 0

                    if gp['channel_dependencies'][j]:
                        channel['component_addr'] = gp['channel_dependencies'][j][0].address

                for channel, next_channel in pair(gp['channels']):
                    channel['next_ch_addr'] = next_channel.address
                next_channel['next_ch_addr'] = 0

                # channel group
                gp['channel_group'].address = address
                gp['channel_group']['first_ch_addr'] = gp['channels'][0].address
                gp['channel_group']['next_cg_addr'] = 0
                for key in ('acq_name_addr', 'comment_addr'):
                    if key in gp['texts']['channel_group'][0]:
                        gp['channel_group'][key] = gp['texts']['channel_group'][0][key].address
                gp['channel_group']['acq_source_addr'] = 0

                gp['data_group']['first_cg_addr'] = address

                address += gp['channel_group']['block_len']
                blocks.append(gp['channel_group'])

            for gp in self.groups:
                for dep_list in gp['channel_dependencies']:
                    if dep_list:
                        if all(isinstance(dep, ChannelArrayBlock) for dep in dep_list):
                            for dep in dep_list:
                                for i, (ch_nr, gp_nr) in enumerate(dep.referenced_channels):
                                    grp = self.groups[gp_nr]
                                    ch = grp['channels'][ch_nr]
                                    dep['scale_axis_{}_dg_addr'.format(i)] = grp['data_group'].address
                                    dep['scale_axis_{}_cg_addr'.format(i)] = grp['channel_group'].address
                                    dep['scale_axis_{}_ch_addr'.format(i)] = ch.address

            for block in blocks:
                write(bytes(block))

            if self.groups:
                self.header['first_dg_addr'] = self.groups[0]['data_group'].address
            else:
                self.header['first_dg_addr'] = 0
            self.header['file_history_addr'] = self.file_history[0][0].address
            self.header['first_attachment_addr'] = self.attachments[0][0].address if self.attachments else 0
            self.header['comment_addr'] = self.file_comment.address if self.file_comment else 0

            seek(v4c.IDENTIFICATION_BLOCK_SIZE , v4c.SEEK_START)
            write(bytes(self.header))

            for orig_addr, gp in zip(original_data_addresses, self.groups):
                gp['data_group']['data_block_addr'] = orig_addr


if __name__ == '__main__':
    pass

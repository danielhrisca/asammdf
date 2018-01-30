# -*- coding: utf-8 -*-
"""
ASAM MDF version 4 file format module
"""

from __future__ import division, print_function

import os
import re
import sys
import time
import warnings
from collections import defaultdict
from copy import deepcopy
from functools import partial, reduce
from hashlib import md5
from math import ceil
from struct import unpack, unpack_from
from tempfile import TemporaryFile

from numexpr import evaluate
from numpy import (
    arange,
    argwhere,
    array,
    array_equal,
    clip,
    dtype,
    flip,
    float64,
    frombuffer,
    interp,
    linspace,
    ones,
    packbits,
    roll,
    searchsorted,
    transpose,
    uint8,
    union1d,
    unpackbits,
    zeros,
)
from numpy.core.defchararray import encode
from numpy.core.records import fromarrays, fromstring

from . import v4_constants as v4c
from .signal import Signal
from .utils import (
    MdfException,
    as_non_byte_sized_signed_int,
    fix_dtype_fields,
    fmt_to_datatype_v4,
    get_fmt_v4,
    get_min_max,
    get_unique_name,
    get_text_v4,
)
from .v4_blocks import (
    AttachmentBlock,
    Channel,
    ChannelArrayBlock,
    ChannelConversion,
    ChannelGroup,
    DataBlock,
    DataGroup,
    DataList,
    DataZippedBlock,
    FileHistory,
    FileIdentificationBlock,
    HeaderBlock,
    HeaderList,
    SignalDataBlock,
    SourceInformation,
    TextBlock,
)
from .version import __version__


MASTER_CHANNELS = (
    v4c.CHANNEL_TYPE_MASTER,
    v4c.CHANNEL_TYPE_VIRTUAL_MASTER,
)

TX = re.compile('<TX>(?P<text>(.|\n)+?)</TX>')

PYVERSION = sys.version_info[0]
if PYVERSION == 2:
    # pylint: disable=W0622
    from .utils import bytes
    # pylint: enable=W0622

__all__ = ['MDF4', ]


class MDF4(object):
    """If the *name* exist it will be memorised otherwise an empty file will be
    created that can be later saved to disk

    Parameters
    ----------
    name : string
        mdf file name
    memory : str
        memory optimization option; default `full`

        * if *full* the data group binary data block will be memorised in RAM
        * if *low* the channel data is read from disk on request, and the
            metadata is memorized into RAM
        * if *minimum* only minimal data is memorized into RAM

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
    memory : str
        memory optimization option
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
    _split_data_blocks = False
    _split_threshold = 1 << 21
    _overwrite = False

    def __init__(self, name=None, memory='full', version='4.10'):
        self.groups = []
        self.header = None
        self.identification = None
        self.file_history = []
        self.name = name
        self.memory = memory
        self.channels_db = {}
        self.masters_db = {}
        self.attachments = []
        self.file_comment = None

        self._ch_map = {}
        self._master_channel_cache = {}
        self._si_map = {}
        self._cc_map = {}

        # used for appending when memory=False
        self._tempfile = TemporaryFile()
        self._file = None

        self._tempfile.write(b'\0')

        if name:
            self._file = open(self.name, 'rb')
            self._read()

        else:
            self.header = HeaderBlock()
            self.identification = FileIdentificationBlock(version=version)
            self.version = version

    def _check_finalised(self):
        flags = self.identification['unfinalized_standard_flags']
        if flags & 1:
            message = ('Unfinalised file {}:'
                       'Update of cycle counters for CG/CA blocks required')
            warnings.warn(message.format(self.name))
        elif flags & 1 << 1:
            message = ('Unfinalised file {}:'
                       'Update of cycle counters for SR blocks required')
            warnings.warn(message.format(self.name))
        elif flags & 1 << 2:
            message = ('Unfinalised file {}:'
                       'Update of length for last DT block required')
            warnings.warn(message.format(self.name))
        elif flags & 1 << 3:
            message = ('Unfinalised file {}:'
                       'Update of length for last RD block required')
            warnings.warn(message.format(self.name))
        elif flags & 1 << 4:
            message = ('Unfinalised file {}:'
                       'Update of last DL block in each chained list'
                       'of DL blocks required')
            warnings.warn(message.format(self.name))
        elif flags & 1 << 5:
            message = ('Unfinalised file {}:'
                       'Update of cg_data_bytes and cg_inval_bytes '
                       'in VLSD CG block required')
            warnings.warn(message.format(self.name))
        elif flags & 1 << 6:
            message = ('Unfinalised file {}:'
                       'Update of offset values for VLSD channel required '
                       'in case a VLSD CG block is used')
            warnings.warn(message.format(self.name))

    def _read(self):
        stream = self._file
        memory = self.memory
        dg_cntr = 0

        self.identification = FileIdentificationBlock(stream=stream)
        version = self.identification['version_str']
        self.version = version.decode('utf-8').strip(' \n\t\0')

        if self.version in ('4.10', '4.11'):
            self._check_finalised()

        self.header = HeaderBlock(address=0x40, stream=stream)

        # read file comment
        if self.header['comment_addr']:
            self.file_comment = TextBlock(
                address=self.header['comment_addr'],
                stream=stream,
            )

        # read file history
        fh_addr = self.header['file_history_addr']
        while fh_addr:
            history_block = FileHistory(address=fh_addr, stream=stream)
            history_text = TextBlock(
                address=history_block['comment_addr'],
                stream=stream,
            )
            self.file_history.append((history_block, history_text))
            fh_addr = history_block['next_fh_addr']

        # read attachments
        at_addr = self.header['first_attachment_addr']
        while at_addr:
            texts = {}
            at_block = AttachmentBlock(address=at_addr, stream=stream)
            for key in ('file_name_addr', 'mime_addr', 'comment_addr'):
                addr = at_block[key]
                if addr:
                    texts[key] = TextBlock(address=addr, stream=stream)

            self.attachments.append((at_block, texts))
            at_addr = at_block['next_at_addr']

        # go to first date group and read each data group sequentially
        dg_addr = self.header['first_dg_addr']

        while dg_addr:
            new_groups = []
            group = DataGroup(address=dg_addr, stream=stream)
            record_id_nr = group['record_id_len']

            # go to first channel group of the current data group
            cg_addr = group['first_cg_addr']

            cg_nr = 0

            cg_size = {}

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
                # channel_group is lsit to allow uniform handling of all texts
                # in save method
                grp['texts'] = {
                    'conversion_tab': [],
                    'channel_group': [],
                }

                # read each channel group sequentially
                block = ChannelGroup(address=cg_addr, stream=stream)
                channel_group = grp['channel_group'] = block

                grp['record_size'] = cg_size

                if channel_group['flags'] == 0:
                    samples_size = channel_group['samples_byte_nr']
                    inval_size = channel_group['invalidation_bytes_nr']
                    record_id = channel_group['record_id']
                    if PYVERSION == 2:
                        record_id = chr(record_id)
                    cg_size[record_id] = samples_size + inval_size
                else:
                    # VLDS flags
                    record_id = channel_group['record_id']
                    if PYVERSION == 2:
                        record_id = chr(record_id)
                    cg_size[record_id] = 0

                if record_id_nr:
                    grp['sorted'] = False
                else:
                    grp['sorted'] = True

                data_group = DataGroup(address=dg_addr, stream=stream)
                grp['data_group'] = data_group

                # read acquisition name and comment for current channel group
                channel_group_texts = {}

                for key in ('acq_name_addr', 'comment_addr'):
                    address = channel_group[key]
                    if address:
                        block = TextBlock(address=address, stream=stream)
                        if memory == 'minimum':
                            channel_group_texts[key] = address
                        else:
                            channel_group_texts[key] = block

                if channel_group_texts:
                    grp['texts']['channel_group'].append(channel_group_texts)
                else:
                    grp['texts']['channel_group'].append(None)

                # go to first channel of the current channel group
                ch_addr = channel_group['first_ch_addr']
                ch_cntr = 0

                # Read channels by walking recursively in the channel group
                # starting from the first channel
                self._read_channels(
                    ch_addr,
                    grp,
                    stream,
                    dg_cntr,
                    ch_cntr,
                )

                cg_addr = channel_group['next_cg_addr']
                dg_cntr += 1

            # store channel groups record sizes dict in each
            # new group data belong to the initial unsorted group, and add
            # the key 'sorted' with the value False to use a flag;
            # this is used later if memory=False

            if memory == 'full':
                grp['data_location'] = v4c.LOCATION_MEMORY
                dat_addr = group['data_block_addr']

                if record_id_nr == 0:
                    size = channel_group['samples_byte_nr']
                    size *= channel_group['cycles_nr']
                else:
                    size = sum(
                        (gp['channel_group']['samples_byte_nr'] + record_id_nr)
                        * gp['channel_group']['cycles_nr']
                        for gp in new_groups
                    )

                data = self._read_data_block(
                    address=dat_addr,
                    stream=stream,
                    size=size,
                )

                if record_id_nr == 0:
                    grp = new_groups[0]
                    grp['data_location'] = v4c.LOCATION_MEMORY
                    grp['data_block'] = DataBlock(data=data)
                else:
                    cg_data = defaultdict(list)

                    i = 0
                    size = len(data)
                    while i < size:
                        rec_id = data[i]
                        # skip record id
                        i += 1
                        rec_size = cg_size[rec_id]
                        if rec_size:
                            rec_data = data[i: i + rec_size]
                            cg_data[rec_id].append(rec_data)
                        else:
                            rec_size = unpack('<I', data[i: i + 4])[0]
                            i += 4
                            rec_data = data[i: i + rec_size]
                            cg_data[rec_id].append(rec_data)
                        # if 2 record id's are used skip also the second one
                        if record_id_nr == 2:
                            i += rec_size + 1
                        else:
                            i += rec_size
                    for grp in new_groups:
                        grp['data_location'] = v4c.LOCATION_MEMORY
                        record_id = grp['channel_group']['record_id']
                        if PYVERSION == 2:
                            record_id = chr(record_id)
                        data = b''.join(cg_data[record_id])
                        grp['channel_group']['record_id'] = 1
                        grp['data_block'] = DataBlock(data=data)
            else:
                for grp in new_groups:
                    grp['data_location'] = v4c.LOCATION_ORIGINAL_FILE

            self.groups.extend(new_groups)

            dg_addr = group['next_dg_addr']

        for grp in self.groups:
            for dep_list in grp['channel_dependencies']:
                if not dep_list:
                    continue

                for dep in dep_list:
                    if isinstance(dep, ChannelArrayBlock):
                        conditions = (
                            dep['ca_type'] == v4c.CA_TYPE_LOOKUP,
                            dep['links_nr'] == 4 * dep['dims'] + 1,
                        )
                        if not all(conditions):
                            continue

                        for i in range(dep['dims']):
                            ch_addr = dep['scale_axis_{}_ch_addr'.format(i)]
                            ref_channel = self._ch_map[ch_addr]
                            dep.referenced_channels.append(ref_channel)
                    else:
                        break

        if self.memory == 'full':
            self.close()

        self._si_map = None
        self._ch_map = None
        self._cc_map = None

    def _read_channels(
            self,
            ch_addr,
            grp,
            stream,
            dg_cntr,
            ch_cntr,
            channel_composition=False):

        memory = self.memory
        channels = grp['channels']
        composition = []
        while ch_addr:
            # read channel block and create channel object
            channel = Channel(address=ch_addr, stream=stream)

            self._ch_map[ch_addr] = (ch_cntr, dg_cntr)

            if memory == 'minimum':
                value = ch_addr
            else:
                value = channel
            channels.append(value)
            if channel_composition:
                composition.append(value)

            # read conversion block and create channel conversion object
            address = channel['conversion_addr']
            if address:
                if memory == 'minimum':
                    conv = ChannelConversion(
                        address=address,
                        stream=stream,
                    )
                    conv_type = conv['conversion_type']
                else:
                    stream.seek(address+8)
                    size = unpack('<Q', stream.read(8))[0]
                    stream.seek(address)
                    raw_bytes = stream.read(size)
                    if raw_bytes in self._cc_map:
                        conv = self._cc_map[raw_bytes]
                        conv_type = conv['conversion_type']
                    else:
                        conv = ChannelConversion(raw_bytes=raw_bytes)
                        conv_type = conv['conversion_type']
                        if conv_type not in v4c.CONVERSIONS_WITH_TEXTS:
                            self._cc_map[raw_bytes] = conv
            else:
                conv_type = -1
                conv = None
            if memory == 'minimum':
                grp['channel_conversions'].append(address)
            else:
                grp['channel_conversions'].append(conv)

            conv_tabx_texts = {}
            # read text fields for channel conversions
            if conv:
                if memory != 'minimum':
                    address = conv['name_addr']
                    if address:
                        conv.name = get_text_v4(address, stream)

                    address = conv['unit_addr']
                    if address:
                        conv.unit = get_text_v4(address, stream)

                    address = conv['comment_addr']
                    if address:
                        conv.comment = get_text_v4(address, stream)

                    address = conv.get('formula_addr', 0)
                    if address:
                        conv.formula = get_text_v4(address, stream)

                if conv_type in v4c.TABULAR_CONVERSIONS:
                    if conv_type == v4c.CONVERSION_TYPE_TTAB:
                        tabs = conv['links_nr'] - 4
                    else:
                        tabs = conv['links_nr'] - 4 - 1
                    for i in range(tabs):
                        address = conv['text_{}'.format(i)]
                        if memory == 'minimum':
                            conv_tabx_texts['text_{}'.format(i)] = address
                        else:
                            if address:
                                block = TextBlock(
                                    address=address,
                                    stream=stream,
                                )
                                conv_tabx_texts['text_{}'.format(i)] = block
                            else:
                                conv_tabx_texts['text_{}'.format(i)] = None
                    if conv_type != v4c.CONVERSION_TYPE_TTAB:
                        address = conv.get('default_addr', 0)
                        if address:
                            if memory == 'minimum':
                                conv_tabx_texts['default_addr'] = address
                            else:
                                stream.seek(address, v4c.SEEK_START)
                                blk_id = stream.read(4)

                                if blk_id == b'##TX':
                                    block = TextBlock(
                                        address=address,
                                        stream=stream,
                                    )
                                    conv_tabx_texts['default_addr'] = block
                                elif blk_id == b'##CC':
                                    block = ChannelConversion(
                                        address=address,
                                        stream=stream,
                                    )
                                    text = str(time.clock()).encode('utf-8')
                                    default_text = block
                                    default_text['text'] = text

                                    conv['unit_addr'] = default_text['unit_addr']
                                    default_text['unit_addr'] = 0

                elif conv_type == v4c.CONVERSION_TYPE_TRANS:
                    # link_nr - common links (4) - default text link (1)
                    for i in range((conv['links_nr'] - 4 - 1) // 2):
                        for key in ('input_{}_addr'.format(i),
                                    'output_{}_addr'.format(i)):
                            address = conv[key]
                            if address:
                                if memory == 'minimum':
                                    conv_tabx_texts[key] = address
                                else:
                                    block = TextBlock(
                                        address=address,
                                        stream=stream,
                                    )
                                    conv_tabx_texts[key] = block
                    address = conv['default_addr']
                    if address:
                        if memory == 'minimum':
                            conv_tabx_texts['default_addr'] = address
                        else:
                            block = TextBlock(
                                address=address,
                                stream=stream,
                            )
                            conv_tabx_texts['default_addr'] = block

            if conv_tabx_texts:
                grp['texts']['conversion_tab'].append(conv_tabx_texts)
            else:
                grp['texts']['conversion_tab'].append(None)

            # read source block and create source information object
            address = channel['source_addr']
            if address:
                if memory == 'minimum':
                    grp['channel_sources'].append(address)
                else:
                    stream.seek(address, v4c.SEEK_START)
                    raw_bytes = stream.read(v4c.SI_BLOCK_SIZE)
                    if raw_bytes in self._si_map:
                        grp['channel_sources'].append(self._si_map[raw_bytes])
                    else:
                        source = SourceInformation(
                            raw_bytes=raw_bytes,
                        )
                        grp['channel_sources'].append(source)

                        address = source['name_addr']
                        if address:
                            source.name = get_text_v4(address, stream)
                        address = source['path_addr']
                        if address:
                            source.path = get_text_v4(address, stream)
                        address = source['comment_addr']
                        if address:
                            source.comment = get_text_v4(address, stream)
                        self._si_map[raw_bytes] = source

            else:
                if memory == 'minimum':
                    grp['channel_sources'].append(0)
                else:
                    grp['channel_sources'].append(None)

            if memory != 'minimum':
                address = channel['unit_addr']
                if address:
                    channel.unit = get_text_v4(address, stream)

                address = channel['comment_addr']
                if address:
                    block = TextBlock(
                        address=address,
                        stream=stream,
                    )
                    name = (
                        block['text']
                        .decode('utf-8')
                        .split('\\')[0]
                        .strip(' \t\n\r\0')
                    )
                    channel.comment = name
                    channel.comment_type = block['id']

            channel.name = name = get_text_v4(channel['name_addr'], stream)

            if name not in self.channels_db:
                self.channels_db[name] = []
            self.channels_db[name].append((dg_cntr, ch_cntr))

            if channel['channel_type'] in MASTER_CHANNELS:
                self.masters_db[dg_cntr] = ch_cntr

            ch_cntr += 1

            if channel['component_addr']:
                # check if it is a CABLOCK or CNBLOCK
                stream.seek(channel['component_addr'], v4c.SEEK_START)
                blk_id = stream.read(4)
                if blk_id == b'##CN':
                    index = ch_cntr - 1
                    grp['channel_dependencies'].append(None)
                    ch_cntr, composition = self._read_channels(
                        channel['component_addr'],
                        grp,
                        stream,
                        dg_cntr,
                        ch_cntr,
                        True,
                    )
                    grp['channel_dependencies'][index] = composition
                else:
                    # only channel arrays with storage=CN_TEMPLATE are
                    # supported so far
                    ca_block = ChannelArrayBlock(
                        address=channel['component_addr'],
                        stream=stream,
                    )
                    if ca_block['storage'] != v4c.CA_STORAGE_TYPE_CN_TEMPLATE:
                        warnings.warn('Only CN template arrays are supported')
                    ca_list = [ca_block, ]
                    while ca_block['composition_addr']:
                        ca_block = ChannelArrayBlock(
                            address=ca_block['composition_addr'],
                            stream=stream,
                        )
                        ca_list.append(ca_block)
                    grp['channel_dependencies'].append(ca_list)

            else:
                grp['channel_dependencies'].append(None)

            # go to next channel of the current channel group
            ch_addr = channel['next_ch_addr']

        return ch_cntr, composition

    def _read_data_block(self, address, stream, size=-1):
        """read and aggregate data blocks for a given data group

        Returns
        -------
        data : bytes
            aggregated raw data
        """
        orig = address
        if address:
            stream.seek(address, v4c.SEEK_START)
            id_string = stream.read(4)
            # can be a DataBlock
            if id_string == b'##DT':
                data = DataBlock(address=address, stream=stream)
                data = data['data']
            # or a DataZippedBlock
            elif id_string == b'##DZ':
                data = DataZippedBlock(address=address, stream=stream)
                data = data['data']
            # or a DataList
            elif id_string == b'##DL':
                if size >= 0:
                    data = bytearray(size)
                    view = memoryview(data)
                    position = 0
                    while address:
                        dl = DataList(address=address, stream=stream)
                        for i in range(dl['links_nr'] - 1):
                            addr = dl['data_block_addr{}'.format(i)]
                            stream.seek(addr, v4c.SEEK_START)
                            id_string = stream.read(4)
                            if id_string == b'##DT':
                                _, dim, __ = unpack('<4s2Q', stream.read(20))
                                dim -= 24
                                position += stream.readinto(view[position: position+dim])
                            elif id_string == b'##DZ':
                                block = DataZippedBlock(
                                    stream=stream,
                                    address=addr,
                                )
                                uncompressed_size = block['original_size']
                                view[position: position+uncompressed_size] = block['data']
                                position += uncompressed_size
                        address = dl['next_dl_addr']

                else:

                    data = []
                    while address:
                        dl = DataList(address=address, stream=stream)
                        for i in range(dl['links_nr'] - 1):
                            addr = dl['data_block_addr{}'.format(i)]
                            stream.seek(addr, v4c.SEEK_START)
                            id_string = stream.read(4)
                            if id_string == b'##DT':
                                block = DataBlock(stream=stream, address=addr)
                                data.append(block['data'])
                            elif id_string == b'##DZ':
                                block = DataZippedBlock(
                                    stream=stream,
                                    address=addr,
                                )
                                data.append(block['data'])
                        address = dl['next_dl_addr']
                    data = b''.join(data)
            # or a header list
            elif id_string == b'##HL':
                hl = HeaderList(address=address, stream=stream)
                data = self._read_data_block(
                    address=hl['first_dl_addr'],
                    stream=stream,
                )
        else:
            data = b''

        return data

    def _load_group_data(self, group):
        """ get group's data block bytes """
        if self.memory == 'full':
            data = group['data_block']['data']
        else:
            # could be an appended group
            # for now appended groups keep the measured data in the memory.
            # the plan is to use a temp file for appended groups, to keep the
            # memory usage low.
            data_group = group['data_group']
            channel_group = group['channel_group']

            if group['data_location'] == v4c.LOCATION_ORIGINAL_FILE:
                # go to the first data block of the current data group
                stream = self._file

                dat_addr = data_group['data_block_addr']
                data = self._read_data_block(
                    address=dat_addr,
                    stream=stream,
                )

                if not group['sorted']:
                    cg_data = []
                    cg_size = group['record_size']
                    record_id = channel_group['record_id']
                    if PYVERSION == 2:
                        record_id = chr(record_id)
                    if data_group['record_id_len'] <= 2:
                        record_id_nr = data_group['record_id_len']
                    else:
                        record_id_nr = 0
                    i = 0
                    size = len(data)
                    while i < size:
                        rec_id = data[i]
                        # skip record id
                        i += 1
                        rec_size = cg_size[rec_id]
                        if rec_size:
                            if rec_id == record_id:
                                rec_data = data[i: i + rec_size]
                                cg_data.append(rec_data)
                        else:
                            rec_size = unpack('<I', data[i: i + 4])[0]
                            i += 4
                            if rec_id == record_id:
                                rec_data = data[i: i + rec_size]
                                cg_data.append(rec_data)
                        # consider the second record ID byte
                        if record_id_nr == 2:
                            i += rec_size + 1
                        else:
                            i += rec_size
                    data = b''.join(cg_data)
            elif group['data_location'] == v4c.LOCATION_TEMPORARY_FILE:

                dat_addr = data_group['data_block_addr']
                if dat_addr:
                    cycles_nr = channel_group['cycles_nr']
                    samples_byte_nr = channel_group['samples_byte_nr']

                    size = cycles_nr * samples_byte_nr

                    self._tempfile.seek(dat_addr, v4c.SEEK_START)
                    data = self._tempfile.read(size)
                else:
                    data = b''

        return data

    def _load_signal_data(self, address):
        """ this method is used to get the channel signal data, usually for
        VLSD channels

        """

        if address:
            if self._file.closed:
                self._file = open(self.name, 'rb')
            stream = self._file
            stream.seek(address, v4c.SEEK_START)
            blk_id = stream.read(4)
            if blk_id == b'##SD':
                data = SignalDataBlock(address=address, stream=stream)
                data = data['data']
            elif blk_id == b'##DZ':
                data = DataZippedBlock(address=address, stream=stream)
                data = data['data']
            elif blk_id == b'##DL':
                data = []
                while address:
                    # the data list will contain only links to SDBLOCK's
                    data_list = DataList(address=address, stream=stream)
                    nr = data_list['links_nr']
                    # aggregate data from all SDBLOCK
                    for i in range(nr - 1):
                        addr = data_list['data_block_addr{}'.format(i)]
                        stream.seek(addr, v4c.SEEK_START)
                        blk_id = stream.read(4)
                        if blk_id == b'##SD':
                            block = SignalDataBlock(
                                address=addr,
                                stream=stream,
                            )
                            data.append(block['data'])
                        elif blk_id == b'##DZ':
                            block = DataZippedBlock(
                                address=addr,
                                stream=stream,
                            )
                            data.append(block['data'])
                        else:
                            message = ('Expected SD, DZ or DL block at {} '
                                       'but found id="{}"')
                            message = message.format(hex(address), blk_id)
                            warnings.warn(message)
                            return b''
                    address = data_list['next_dl_addr']
                data = b''.join(data)
            elif blk_id == b'##CN':
                data = b''
            else:
                message = ('Expected SD, DL, DZ or CN block at {} '
                           'but found id="{}"')
                message = message.format(hex(address), blk_id)
                warnings.warn(message)
                data = b''
            if self.memory == 'full':
                self.close()
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
            mapping of channels to records fields, records fields dtype

        """
        grp = group
        stream = self._file
        memory = self.memory
        channel_group = grp['channel_group']
        if memory == 'minimum':
            channels = [
                Channel(address=ch_addr, stream=stream)
                for ch_addr in grp['channels']
            ]
        else:
            channels = grp['channels']

        record_size = channel_group['samples_byte_nr']
        invalidation_bytes_nr = channel_group['invalidation_bytes_nr']
        next_byte_aligned_position = 0
        types = []
        current_parent = ""
        parent_start_offset = 0
        parents = {}
        group_channels = set()

        sortedchannels = sorted(enumerate(channels), key=lambda i: i[1])
        for original_index, new_ch in sortedchannels:

            start_offset = new_ch['byte_offset']
            bit_offset = new_ch['bit_offset']
            data_type = new_ch['data_type']
            bit_count = new_ch['bit_count']
            ch_type = new_ch['channel_type']
            dependency_list = grp['channel_dependencies'][original_index]
            if memory == 'minimum':
                block = TextBlock(
                    address=new_ch['name_addr'],
                    stream=stream,
                )
                name = (
                    block['text']
                    .decode('utf-8')
                    .strip(' \r\n\t\0')
                    .split('\\')[0]
                )
            else:
                name = new_ch.name

            # handle multiple occurance of same channel name
            name = get_unique_name(group_channels, name)
            group_channels.add(name)

            if start_offset >= next_byte_aligned_position:
                if ch_type not in (v4c.CHANNEL_TYPE_VIRTUAL_MASTER,
                                   v4c.CHANNEL_TYPE_VIRTUAL):
                    if not dependency_list:
                        parent_start_offset = start_offset

                        # check if there are byte gaps in the record
                        gap = parent_start_offset - next_byte_aligned_position
                        if gap:
                            types.append(('', 'a{}'.format(gap)))

                        # adjust size to 1, 2, 4 or 8 bytes
                        size = bit_offset + bit_count
                        if data_type not in (v4c.DATA_TYPE_BYTEARRAY,
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
                            dtype_pair = name, get_fmt_v4(data_type, size)
                            types.append(dtype_pair)
                            parents[original_index] = name, bit_offset

                        current_parent = name
                    else:
                        if isinstance(dependency_list[0], ChannelArrayBlock):
                            ca_block = dependency_list[0]

                            # check if there are byte gaps in the record
                            gap = start_offset - next_byte_aligned_position
                            if gap:
                                dtype_pair = '', 'a{}'.format(gap)
                                types.append(dtype_pair)

                            size = bit_count >> 3
                            shape = tuple(
                                ca_block['dim_size_{}'.format(i)]
                                for i in range(ca_block['dims'])
                            )

                            if ca_block['byte_offset_base'] // size > 1 and \
                                    len(shape) == 1:
                                shape += ca_block['byte_offset_base'] // size,
                            dim = 1
                            for d in shape:
                                dim *= d

                            dtype_pair = name, get_fmt_v4(data_type, size), shape
                            types.append(dtype_pair)

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
                    parents[original_index] = current_parent, ((start_offset - parent_start_offset) << 3) + bit_offset
            if next_byte_aligned_position > record_size:
                break

        gap = record_size - next_byte_aligned_position
        if gap > 0:
            dtype_pair = '', 'a{}'.format(gap)
            types.append(dtype_pair)

        dtype_pair = 'invalidation_bytes', 'u1', invalidation_bytes_nr
        types.append(dtype_pair)
        if PYVERSION == 2:
            types = fix_dtype_fields(types)

        return parents, dtype(types)

    def _get_not_byte_aligned_data(self, data, group, ch_nr):
        big_endian_types = (
            v4c.DATA_TYPE_UNSIGNED_MOTOROLA,
            v4c.DATA_TYPE_REAL_MOTOROLA,
            v4c.DATA_TYPE_SIGNED_MOTOROLA,
        )

        record_size = group['channel_group']['samples_byte_nr']

        if self.memory == 'minimum':
            channel = Channel(
                address=group['channels'][ch_nr],
                stream=self._file,
            )
        else:
            channel = group['channels'][ch_nr]

        bit_offset = channel['bit_offset']
        byte_offset = channel['byte_offset']
        bit_count = channel['bit_count']

        dependencies = group['channel_dependencies'][ch_nr]
        if dependencies and isinstance(dependencies[0], ChannelArrayBlock):
            ca_block = dependencies[0]

            size = bit_count >> 3
            shape = tuple(
                ca_block['dim_size_{}'.format(i)]
                for i in range(ca_block['dims'])
            )
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

        types = [
            ('', 'a{}'.format(byte_offset)),
            ('vals', '({},)u1'.format(byte_count)),
            ('', 'a{}'.format(record_size - byte_count - byte_offset)),
        ]

        vals = fromstring(data, dtype=dtype(types))

        vals.setflags(write=False)

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
        else:
            size = bit_count // 8

        if size > byte_count:
            extra_bytes = size - byte_count
            extra = zeros((len(vals), extra_bytes), dtype=uint8)

            types = [
                ('vals', vals.dtype, vals.shape[1:]),
                ('', extra.dtype, extra.shape[1:]),
            ]
            vals = fromarrays([vals, extra], dtype=dtype(types))
        vals = vals.tostring()

        fmt = get_fmt_v4(channel['data_type'], size)
        if size <= byte_count:
            types = [
                ('vals', fmt),
                ('', 'a{}'.format(byte_count - size)),
            ]
        else:
            types = [('vals', fmt), ]

        vals = fromstring(vals, dtype=dtype(types))

        return vals['vals']

    def _validate_channel_selection(self, name=None, group=None, index=None):
        """Gets channel comment.
        Channel can be specified in two ways:

        * using the first positional argument *name*

            * if there are multiple occurrences for this channel then the
            *group* and *index* arguments can be used to select a specific
            group.
            * if there are multiple occurrences for this channel and either the
            *group* or *index* arguments is None then a warning is issued

        * using the group number (keyword argument *group*) and the channel
        number (keyword argument *index*). Use *info* method for group and
        channel numbers


        If the *raster* keyword argument is not *None* the output is
        interpolated accordingly.

        Parameters
        ----------
        name : string
            name of channel
        group : int
            0-based group index
        index : int
            0-based channel index

        Returns
        -------
        group_index, channel_index : (int, int)
            selected channel's group and channel index

        """
        if name is None:
            if group is None or index is None:
                message = (
                    'Invalid arguments for channel selection: '
                    'must give "name" or, "group" and "index"'
                )
                raise MdfException(message)
            else:
                gp_nr, ch_nr = group, index
                if gp_nr > len(self.groups) - 1:
                    raise MdfException('Group index out of range')
                if index > len(self.groups[gp_nr]['channels']) - 1:
                    raise MdfException('Channel index out of range')
        else:
            name = name.split('\\')[0]
            if name not in self.channels_db:
                raise MdfException('Channel "{}" not found'.format(name))
            else:
                if group is None:
                    gp_nr, ch_nr = self.channels_db[name][0]
                    if len(self.channels_db[name]) > 1:
                        message = (
                            'Multiple occurances for channel "{}". '
                            'Using first occurance from data group {}. '
                            'Provide both "group" and "index" arguments'
                            ' to select another data group'
                        )
                        message = message.format(name, gp_nr)
                        warnings.warn(message)
                else:
                    for gp_nr, ch_nr in self.channels_db[name]:
                        if gp_nr == group:
                            if index is None:
                                break
                            elif index == ch_nr:
                                break
                    else:
                        if index is None:
                            message = 'Channel "{}" not found in group {}'
                            message = message.format(name, group)
                        else:
                            message = (
                                'Channel "{}" not found in group {} '
                                'at index {}'
                            )
                            message = message.format(name, group, index)
                        raise MdfException(message)

        return gp_nr, ch_nr

    def append(self, signals, source_info='Python', common_timebase=False):
        """
        Appends a new data group.

        For channel dependencies type Signals, the *samples* attribute must be
        a numpy.recarray

        Parameters
        ----------
        signals : list
            list on *Signal* objects
        source_info : str
            source information; default 'Python'
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
            message = '"append" requires a non-empty list of Signal objects'
            raise MdfException(message)

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

        # split regular from composed signals. Composed signals have recarray
        # samples or multimendional ndarray.
        # The regular signals will be first added to the group.
        # The composed signals will be saved along side the fields, which will
        # be saved as new signals.
        simple_signals = [
            sig for sig in signals
            if len(sig.samples.shape) <= 1
            and sig.samples.dtype.names is None
        ]
        composed_signals = [
            sig for sig in signals
            if len(sig.samples.shape) > 1
            or sig.samples.dtype.names
        ]

        dg_cntr = len(self.groups)

        gp = {}
        gp['channels'] = gp_channels = []
        gp['channel_conversions'] = gp_conv = []
        gp['channel_sources'] = gp_source = []
        gp['channel_dependencies'] = gp_dep = []
        gp['texts'] = gp_texts = {
            'conversion_tab': [],
            'channel_group': [],
        }

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
        for key in ('conversion_tab',):
            gp_texts[key].append(None)

        memory = self.memory
        file = self._tempfile
        write = file.write
        tell = file.tell

        if memory == 'minimum':
            block = TextBlock(text='t', meta=False)
            channel_name_addr = tell()
            write(bytes(block))

        if memory == 'minimum':
            block = TextBlock(text='s', meta=False)
            channel_unit_addr = tell()
            write(bytes(block))

        if memory == 'minimum':
            block = TextBlock(text=source_info, meta=False)
            source_text_address = tell()
            write(bytes(block))
        else:
            source_text_address = 0

        source_block = SourceInformation(
            name_addr=source_text_address,
            path_addr=source_text_address,
        )
        source_block.name = source_block.path = source_info

        source_info_address = tell()
        write(bytes(source_block))

        # conversion and source for time channel
        if memory == 'minimum':
            gp_conv.append(0)
            gp_source.append(source_info_address)
        else:
            gp_conv.append(None)
            gp_source.append(source_block)

        if memory == 'minimum':
            name_addr = channel_name_addr
            unit_addr = channel_unit_addr
        else:
            name_addr = 0
            unit_addr = 0

        # time channel
        t_type, t_size = fmt_to_datatype_v4(t.dtype)
        kargs = {
            'channel_type': v4c.CHANNEL_TYPE_MASTER,
            'data_type': t_type,
            'sync_type': 1,
            'byte_offset': 0,
            'bit_offset': 0,
            'bit_count': t_size,
            'min_raw_value': t[0] if cycles_nr else 0,
            'max_raw_value': t[-1] if cycles_nr else 0,
            'lower_limit': t[0] if cycles_nr else 0,
            'upper_limit': t[-1] if cycles_nr else 0,
            'flags': v4c.FLAG_PHY_RANGE_OK | v4c.FLAG_VAL_RANGE_OK,
            'name_addr': name_addr,
            'unit_addr': unit_addr,
        }
        ch = Channel(**kargs)
        name = 't'
        if memory == 'minimum':
            address = tell()
            write(bytes(ch))
            gp_channels.append(address)
        else:
            ch.name = name
            ch.unit = 's'
            gp_channels.append(ch)

        if name not in self.channels_db:
            self.channels_db[name] = []
        self.channels_db[name].append((dg_cntr, ch_cntr))
        self.masters_db[dg_cntr] = 0
        # data group record parents
        parents[ch_cntr] = name, 0

        # time channel doesn't have channel dependencies
        gp_dep.append(None)

        fields.append(t)
        types.append((name, t.dtype))
        field_names.add(name)

        offset += t_size // 8
        ch_cntr += 1

        if self._compact_integers_on_append:
            compacted_signals = [
                {'signal': sig}
                for sig in simple_signals
                if sig.samples.dtype.kind in 'ui'
            ]

            max_itemsize = 1
            dtype_ = dtype(uint8)

            for signal in compacted_signals:
                itemsize = signal['signal'].samples.dtype.itemsize

                min_val, max_val = get_min_max(signal['signal'].samples)
                signal['min'], signal['max'] = min_val, max_val
                minimum_bitlength = (itemsize // 2) * 8 + 1
                bit_length = max(
                    int(min_val).bit_length(),
                    int(max_val).bit_length(),
                )
                bit_length += 1

                signal['bit_count'] = max(minimum_bitlength, bit_length)

                if itemsize > max_itemsize:
                    dtype_ = dtype('<u{}'.format(itemsize))
                    max_itemsize = itemsize

            compacted_signals.sort(key=lambda x: x['bit_count'])
            simple_signals = [
                sig
                for sig in simple_signals
                if sig.samples.dtype.kind not in 'ui'
            ]
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
            types.append((field_name, dtype_))
            field_names.add(field_name)

            values = zeros(cycles_nr, dtype=dtype_)

            for signal_d in cluster:

                signal = signal_d['signal']
                bit_count = signal_d['bit_count']
                min_val = signal_d['min']
                max_val = signal_d['max']

                name = signal.name
                gp_texts['conversion_tab'].append(None)

                if memory == 'minimum':
                    block = TextBlock(text=name, meta=False)
                    channel_name_address = tell()
                    write(bytes(block))

                    if signal.unit:
                        block = TextBlock(text=signal.unit, meta=False)
                        channel_unit_address = tell()
                        write(bytes(block))
                    else:
                        channel_unit_address = 0

                    if signal.comment:
                        block = TextBlock(text=signal.comment, meta=False)
                        channel_comment_address = tell()
                        write(bytes(block))
                    else:
                        channel_comment_address = 0

                # conversions for channel
                info = signal.info
                conv_texts_tab = {}
                if info and 'raw' in info:
                    kargs = {}
                    raw = info['raw']
                    phys = info['phys']
                    if raw.dtype.kind == 'S':
                        kargs['conversion_type'] = v4c.CONVERSION_TYPE_TTAB
                        for i, (r_, p_) in enumerate(zip(raw, phys)):
                            kargs['text_{}'.format(i)] = 0
                            kargs['val_{}'.format(i)] = p_

                            block = TextBlock(
                                text=r_,
                                meta=False,
                            )
                            if memory != 'minimum':
                                conv_texts_tab['text_{}'.format(i)] = block
                            else:
                                address = tell()
                                conv_texts_tab['text_{}'.format(i)] = address
                                write(bytes(block))
                        kargs['val_default'] = info['default']
                        kargs['links_nr'] = len(raw) + 4
                    else:
                        kargs['conversion_type'] = v4c.CONVERSION_TYPE_TABX
                        for i, (r_, p_) in enumerate(zip(raw, phys)):
                            kargs['text_{}'.format(i)] = 0
                            kargs['val_{}'.format(i)] = r_

                            block = TextBlock(
                                text=p_,
                                meta=False,
                            )
                            if memory != 'minimum':
                                conv_texts_tab['text_{}'.format(i)] = block
                            else:
                                address = tell()
                                conv_texts_tab['text_{}'.format(i)] = address
                                write(bytes(block))
                        if 'default' in info:
                            block = TextBlock(
                                text=info['default'],
                                meta=False,
                            )
                            if memory != 'minimum':
                                conv_texts_tab['default_addr'] = block
                            else:
                                address = tell()
                                conv_texts_tab['default_addr'] = address
                                write(bytes(block))
                        kargs['links_nr'] = len(raw) + 5
                    block = ChannelConversion(**kargs)
                    if memory != 'minimum':
                        gp_conv.append(block)
                    else:
                        address = tell()
                        gp_conv.append(address)
                        write(bytes(block))
                elif info and 'lower' in info:
                    kargs = {}
                    kargs['conversion_type'] = v4c.CONVERSION_TYPE_RTABX
                    lower = info['lower']
                    upper = info['upper']
                    texts = info['phys']
                    kargs['ref_param_nr'] = len(upper)
                    kargs['links_nr'] = len(lower) + 5

                    for i, (u_, l_, t_) in enumerate(zip(upper, lower, texts)):
                        kargs['lower_{}'.format(i)] = l_
                        kargs['upper_{}'.format(i)] = u_
                        kargs['text_{}'.format(i)] = 0

                        block = TextBlock(
                            text=t_,
                            meta=False,
                        )
                        if memory != 'minimum':
                            conv_texts_tab['text_{}'.format(i)] = block
                        else:
                            address = tell()
                            conv_texts_tab['text_{}'.format(i)] = address
                            write(bytes(block))
                    if 'default' in info:
                        block = TextBlock(
                            text=info['default'],
                            meta=False,
                        )
                        if memory != 'minimum':
                            conv_texts_tab['default_addr'] = block
                        else:
                            address = tell()
                            conv_texts_tab['default_addr'] = address
                            write(bytes(block))

                    block = ChannelConversion(**kargs)

                    if memory != 'minimum':
                        gp_conv.append(block)
                    else:
                        address = tell()
                        gp_conv.append(address)
                        write(bytes(block))

                else:
                    if memory != 'minimum':
                        gp_conv.append(None)
                    else:
                        gp_conv.append(0)

                if conv_texts_tab:
                    gp_texts['conversion_tab'][-1] = conv_texts_tab

                # source for channel
                if memory != 'minimum':
                    gp_source.append(source_block)
                else:
                    gp_source.append(source_info_address)

                if memory == 'minimum':
                    name_addr = channel_name_address
                    unit_addr = channel_unit_address
                    comment_addr = channel_comment_address
                else:
                    name_addr = 0
                    unit_addr = 0
                    comment_addr = 0

                # compute additional byte offset for large records size
                if signal.samples.dtype.kind == 'u':
                    data_type = v4c.DATA_TYPE_UNSIGNED_INTEL
                else:
                    data_type = v4c.DATA_TYPE_SIGNED_INTEL
                kargs = {
                    'channel_type': v4c.CHANNEL_TYPE_VALUE,
                    'bit_count': bit_count,
                    'byte_offset': offset + bit_offset // 8,
                    'bit_offset': bit_offset % 8,
                    'data_type': data_type,
                    'min_raw_value': min_val if min_val <= max_val else 0,
                    'max_raw_value': max_val if min_val <= max_val else 0,
                    'lower_limit': min_val if min_val <= max_val else 0,
                    'upper_limit': max_val if min_val <= max_val else 0,
                    'name_addr': name_addr,
                    'unit_addr': unit_addr,
                    'comment_addr': comment_addr,
                }
                if min_val > max_val:
                    kargs['flags'] = 0
                else:
                    kargs['flags'] = v4c.FLAG_PHY_RANGE_OK | v4c.FLAG_VAL_RANGE_OK
                ch = Channel(**kargs)
                if memory != 'minimum':
                    ch.name = name
                    ch.unit = signal.unit
                    ch.comment = signal.comment
                    gp_channels.append(ch)
                else:
                    address = tell()
                    write(bytes(ch))
                    gp_channels.append(address)

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
            name = signal.name
            gp_texts['conversion_tab'].append(None)

            if memory == 'minimum':
                block = TextBlock(text=name, meta=False)
                channel_name_address = tell()
                write(bytes(block))

                if signal.unit:
                    block = TextBlock(
                        text=signal.unit,
                        meta=False,
                    )

                    channel_unit_address = tell()
                    write(bytes(block))
                else:
                    channel_unit_address = 0

                if signal.comment:
                    block = TextBlock(text=signal.comment, meta=False)
                    channel_comment_address = tell()
                    write(bytes(block))
                else:
                    channel_comment_address = 0

            # conversions for channel
            info = signal.info
            conv_texts_tab = {}
            if info and 'raw' in info:
                kargs = {}
                raw = info['raw']
                phys = info['phys']
                if raw.dtype.kind == 'S':
                    kargs['conversion_type'] = v4c.CONVERSION_TYPE_TTAB
                    for i, (r_, p_) in enumerate(zip(raw, phys)):
                        kargs['text_{}'.format(i)] = 0
                        kargs['val_{}'.format(i)] = p_

                        block = TextBlock(
                            text=r_,
                            meta=False,
                        )
                        if memory != 'minimum':
                            conv_texts_tab['text_{}'.format(i)] = block
                        else:
                            address = tell()
                            conv_texts_tab['text_{}'.format(i)] = address
                            write(bytes(block))
                    kargs['val_default'] = info['default']
                    kargs['links_nr'] = len(raw) + 4
                else:
                    kargs['conversion_type'] = v4c.CONVERSION_TYPE_TABX
                    for i, (r_, p_) in enumerate(zip(raw, phys)):
                        kargs['text_{}'.format(i)] = 0
                        kargs['val_{}'.format(i)] = r_

                        block = TextBlock(
                            text=p_,
                            meta=False,
                        )
                        if memory != 'minimum':
                            conv_texts_tab['text_{}'.format(i)] = block
                        else:
                            address = tell()
                            conv_texts_tab['text_{}'.format(i)] = address
                            write(bytes(block))
                    if 'default' in info:
                        block = TextBlock(
                            text=info['default'],
                            meta=False,
                        )
                        if memory != 'minimum':
                            conv_texts_tab['default_addr'] = block
                        else:
                            address = tell()
                            conv_texts_tab['default_addr'] = address
                            write(bytes(block))
                    kargs['links_nr'] = len(raw) + 5

                block = ChannelConversion(**kargs)
                if memory != 'minimum':
                    gp_conv.append(block)
                else:
                    address = tell()
                    gp_conv.append(address)
                    write(bytes(block))

            elif info and 'lower' in info:
                kargs = {}
                kargs['conversion_type'] = v4c.CONVERSION_TYPE_RTABX
                lower = info['lower']
                upper = info['upper']
                texts = info['phys']
                kargs['ref_param_nr'] = len(upper)
                kargs['links_nr'] = len(lower) + 5

                for i, (u_, l_, t_) in enumerate(zip(upper, lower, texts)):
                    kargs['lower_{}'.format(i)] = l_
                    kargs['upper_{}'.format(i)] = u_
                    kargs['text_{}'.format(i)] = 0

                    block = TextBlock(
                        text=t_,
                        meta=False,
                    )
                    if memory != 'minimum':
                        conv_texts_tab['text_{}'.format(i)] = block
                    else:
                        address = tell()
                        conv_texts_tab['text_{}'.format(i)] = address
                        write(bytes(block))
                if 'default' in info:
                    block = TextBlock(
                        text=info['default'],
                        meta=False,
                    )
                    if memory != 'minimum':
                        conv_texts_tab['default_addr'] = block
                    else:
                        address = tell()
                        conv_texts_tab['default_addr'] = address
                        write(bytes(block))
                block = ChannelConversion(**kargs)
                if memory != 'minimum':
                    gp_conv.append(block)
                else:
                    address = tell()
                    gp_conv.append(address)
                    write(bytes(block))

            else:
                if memory != 'minimum':
                    gp_conv.append(None)
                else:
                    gp_conv.append(0)

            if conv_texts_tab:
                gp_texts['conversion_tab'][-1] = conv_texts_tab

            # source for channel
            if memory != 'minimum':
                gp_source.append(source_block)
            else:
                gp_source.append(source_info_address)

            if memory == 'minimum':
                name_addr = channel_name_address
                unit_addr = channel_unit_address
                comment_addr = channel_comment_address
            else:
                name_addr = 0
                unit_addr = 0
                comment_addr = 0

            # compute additional byte offset for large records size
            s_type, s_size = fmt_to_datatype_v4(signal.samples.dtype)
            byte_size = max(s_size // 8, 1)
            min_val, max_val = get_min_max(signal.samples)
            kargs = {
                'channel_type': v4c.CHANNEL_TYPE_VALUE,
                'bit_count': s_size,
                'byte_offset': offset,
                'bit_offset': 0,
                'data_type': s_type,
                'min_raw_value': min_val if min_val <= max_val else 0,
                'max_raw_value': max_val if min_val <= max_val else 0,
                'lower_limit': min_val if min_val <= max_val else 0,
                'upper_limit': max_val if min_val <= max_val else 0,
                'name_addr': name_addr,
                'unit_addr': unit_addr,
                'comment_addr': comment_addr,
            }
            if min_val > max_val:
                kargs['flags'] = 0
            else:
                kargs['flags'] = v4c.FLAG_PHY_RANGE_OK | v4c.FLAG_VAL_RANGE_OK
            ch = Channel(**kargs)
            if memory != 'minimum':
                ch.name = name
                ch.unit = signal.unit
                ch.comment = signal.comment
                gp_channels.append(ch)
            else:
                address = tell()
                write(bytes(ch))
                gp_channels.append(address)

            offset += byte_size

            if name not in self.channels_db:
                self.channels_db[name] = []
            self.channels_db[name].append((dg_cntr, ch_cntr))

            # update the parents as well
            field_name = get_unique_name(field_names, name)
            parents[ch_cntr] = field_name, 0

            fields.append(signal.samples)
            types.append((field_name, signal.samples.dtype))
            field_names.add(field_name)

            ch_cntr += 1

            # simple channels don't have channel dependencies
            gp_dep.append(None)

        canopen_time_fields = (
            'ms',
            'days',
        )
        canopen_date_fields = (
            'ms',
            'min',
            'hour',
            'day',
            'month',
            'year',
            'summer_time',
            'day_of_week',
        )
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
                    types.append((field_name, 'V6'))
                    byte_size = 6
                    s_type = v4c.DATA_TYPE_CANOPEN_TIME

                else:
                    vals = []
                    for field in ('ms', 'min', 'hour', 'day', 'month', 'year'):
                        vals.append(signal.samples[field])
                    vals = fromarrays(vals).tostring()

                    fields.append(frombuffer(vals, dtype='V7'))
                    types.append((field_name, 'V7'))
                    byte_size = 7
                    s_type = v4c.DATA_TYPE_CANOPEN_DATE

                s_size = byte_size << 3

                # add channel texts
                gp_texts['conversion_tab'].append(None)

                if memory == 'minimum':
                    block = TextBlock(text=name, meta=False)
                    channel_name_address = tell()
                    write(bytes(block))
                    if signal.unit:
                        block = TextBlock(
                            text=signal.unit,
                            meta=False,
                        )
                        channel_unit_address = tell()
                        write(bytes(block))

                    if signal.comment:
                        block = TextBlock(text=signal.comment, meta=False)
                        channel_comment_address = tell()
                        write(bytes(block))
                    else:
                        channel_comment_address = 0

                # add channel conversion
                if memory != 'minimum':
                    gp_conv.append(None)
                else:
                    gp_conv.append(0)

                if memory != 'minimum':
                    gp_source.append(source_block)
                else:
                    gp_source.append(source_info_address)

                # there is no channel dependency
                gp_dep.append(None)

                if memory == 'minimum':
                    name_addr = channel_name_address
                    unit_addr = channel_unit_address
                    comment_addr = channel_comment_address
                else:
                    name_addr = 0
                    unit_addr = 0
                    comment_addr = 0

                # add channel block
                kargs = {
                    'channel_type': v4c.CHANNEL_TYPE_VALUE,
                    'bit_count': s_size,
                    'byte_offset': offset,
                    'bit_offset': 0,
                    'data_type': s_type,
                    'min_raw_value': 0,
                    'max_raw_value': 0,
                    'lower_limit': 0,
                    'upper_limit': 0,
                    'flags': 0,
                    'name_addr': name_addr,
                    'unit_addr': unit_addr,
                    'comment_addr': comment_addr,
                }
                ch = Channel(**kargs)
                if memory != 'minimum':
                    ch.name = name
                    ch.unit = signal.unit
                    ch.comment = signal.comment
                    gp_channels.append(ch)
                else:
                    address = tell()
                    write(bytes(ch))
                    gp_channels.append(address)

                offset += byte_size

                if name in self.channels_db:
                    self.channels_db[name].append((dg_cntr, ch_cntr))
                else:
                    self.channels_db[name] = []
                    self.channels_db[name].append((dg_cntr, ch_cntr))

                # update the parents as well
                parents[ch_cntr] = field_name, 0

                ch_cntr += 1

            elif names and names[0] != name:
                # here we have a structure channel composition

                field_name = get_unique_name(field_names, name)
                field_names.add(field_name)

                # first we add the structure channel
                # add channel texts
                gp_texts['conversion_tab'].append(None)

                if memory == 'minimum':
                    block = TextBlock(text=name, meta=False)
                    channel_name_address = tell()
                    write(bytes(block))

                    if signal.unit:
                        block = TextBlock(text=signal.unit, meta=False)
                        channel_unit_address = tell()
                        write(bytes(block))
                    else:
                        channel_unit_address = 0

                    if signal.comment:
                        block = TextBlock(text=signal.comment, meta=False)
                        channel_comment_address = tell()
                        write(bytes(block))
                    else:
                        channel_comment_address = 0

                # add channel conversion
                if memory != 'minimum':
                    gp_conv.append(None)
                else:
                    gp_conv.append(0)

                if memory != 'minimum':
                    gp_source.append(source_block)
                else:
                    gp_source.append(source_info_address)

                if memory == 'minimum':
                    name_addr = channel_name_address
                    unit_addr = channel_unit_address
                    comment_addr = channel_comment_address
                else:
                    name_addr = 0
                    unit_addr = 0
                    comment_addr = 0

                # add channel block
                kargs = {
                    'channel_type': v4c.CHANNEL_TYPE_VALUE,
                    'bit_count': 8,
                    'byte_offset': offset,
                    'bit_offset': 0,
                    'data_type': v4c.DATA_TYPE_BYTEARRAY,
                    'min_raw_value': 0,
                    'max_raw_value': 0,
                    'lower_limit': 0,
                    'upper_limit': 0,
                    'flags': 0,
                    'name_addr': name_addr,
                    'unit_addr': unit_addr,
                    'comment_addr': comment_addr,
                }
                ch = Channel(**kargs)
                if memory != 'minimum':
                    ch.name = name
                    ch.unit = signal.unit
                    ch.comment = signal.comment
                    gp_channels.append(ch)
                else:
                    address = tell()
                    write(bytes(ch))
                    gp_channels.append(address)

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

                    s_type, s_size = fmt_to_datatype_v4(samples.dtype)
                    byte_size = s_size >> 3

                    fields.append(samples)
                    types.append((field_name, samples.dtype))

                    # add channel texts
                    gp_texts['conversion_tab'].append(None)

                    if memory == 'minimum':
                        block = TextBlock(text=name, meta=False)
                        channel_name_address = tell()
                        write(bytes(block))

                        if signal.unit:
                            block = TextBlock(text=signal.unit, meta=False)
                            channel_unit_address = tell()
                            write(bytes(block))
                        else:
                            channel_unit_address = 0

                        if signal.comment:
                            block = TextBlock(text=signal.comment, meta=False)
                            channel_comment_address = tell()
                            write(bytes(block))
                        else:
                            channel_comment_address = 0

                    # add channel conversion
                    if memory != 'minimum':
                        gp_conv.append(None)
                    else:
                        gp_conv.append(0)

                    # source
                    if memory != 'minimum':
                        gp_source.append(source_block)
                    else:
                        gp_source.append(source_info_address)

                    if memory == 'minimum':
                        name_addr = channel_name_address
                        unit_addr = channel_unit_address
                        comment_addr = channel_comment_address
                    else:
                        name_addr = 0
                        unit_addr = 0
                        comment_addr = 0

                    # add channel block
                    min_val, max_val = get_min_max(signal.samples)
                    kargs = {
                        'channel_type': v4c.CHANNEL_TYPE_VALUE,
                        'bit_count': s_size,
                        'byte_offset': offset,
                        'bit_offset': 0,
                        'data_type': s_type,
                        'min_raw_value': min_val if min_val <= max_val else 0,
                        'max_raw_value': max_val if min_val <= max_val else 0,
                        'lower_limit': min_val if min_val <= max_val else 0,
                        'upper_limit': max_val if min_val <= max_val else 0,
                        'flags': v4c.FLAG_PHY_RANGE_OK | v4c.FLAG_VAL_RANGE_OK,
                        'name_addr': name_addr,
                        'unit_addr': unit_addr,
                        'comment_addr': comment_addr,
                    }
                    ch = Channel(**kargs)
                    if memory != 'minimum':
                        ch.name = name
                        ch.unit = signal.unit
                        ch.comment = signal.comment
                        gp_channels.append(ch)
                        dep_list.append(ch)
                    else:
                        address = tell()
                        write(bytes(ch))
                        gp_channels.append(address)
                        dep_list.append(address)

                    offset += byte_size

                    if name not in self.channels_db:
                        self.channels_db[name] = []
                    self.channels_db[name].append((dg_cntr, ch_cntr))

                    # update the parents as well
                    parents[ch_cntr] = field_name, 0

                    ch_cntr += 1
                    gp_dep.append(None)

            else:
                # here we have channel arrays or mdf v3 channel dependencies
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
                        kargs = {
                            'dims': dims_nr,
                            'ca_type': v4c.CA_TYPE_LOOKUP,
                            'flags': v4c.FLAG_CA_FIXED_AXIS,
                            'byte_offset_base': samples.dtype.itemsize,
                        }
                        for i in range(dims_nr):
                            kargs['dim_size_{}'.format(i)] = shape[i]

                    elif len(names) == 1:
                        kargs = {
                            'dims': dims_nr,
                            'ca_type': v4c.CA_TYPE_ARRAY,
                            'flags': 0,
                            'byte_offset_base': samples.dtype.itemsize,
                        }
                        for i in range(dims_nr):
                            kargs['dim_size_{}'.format(i)] = shape[i]

                    else:
                        kargs = {
                            'dims': dims_nr,
                            'ca_type': v4c.CA_TYPE_LOOKUP,
                            'flags': v4c.FLAG_CA_AXIS,
                            'byte_offset_base': samples.dtype.itemsize,
                        }
                        for i in range(dims_nr):
                            kargs['dim_size_{}'.format(i)] = shape[i]

                    parent_dep = ChannelArrayBlock(**kargs)
                    gp_dep.append([parent_dep, ])

                else:
                    # add channel dependency block for composed parent channel
                    kargs = {
                        'dims': 1,
                        'ca_type': v4c.CA_TYPE_SCALE_AXIS,
                        'flags': 0,
                        'byte_offset_base': samples.dtype.itemsize,
                        'dim_size_0': shape[0],
                    }
                    parent_dep = ChannelArrayBlock(**kargs)
                    gp_dep.append([parent_dep, ])

                field_name = get_unique_name(field_names, name)
                field_names.add(field_name)

                fields.append(samples)
                dtype_pair = field_name, samples.dtype, shape
                types.append(dtype_pair)

                # first we add the structure channel
                # add channel texts
                gp_texts['conversion_tab'].append(None)

                if memory == 'minimum':
                    block = TextBlock(text=name, meta=False)
                    channel_name_address = tell()
                    write(bytes(block))

                    if signal.unit:
                        block = TextBlock(text=signal.unit, meta=False)
                        channel_unit_address = tell()
                        write(bytes(block))
                    else:
                        channel_unit_address = 0

                    if signal.comment:
                        block = TextBlock(text=signal.comment, meta=False)
                        channel_comment_address = tell()
                        write(bytes(block))
                    else:
                        channel_comment_address = 0

                # add channel conversion
                if memory != 'minimum':
                    gp_conv.append(None)
                else:
                    gp_conv.append(0)

                # source for channel
                if memory != 'minimum':
                    gp_source.append(source_block)
                else:
                    gp_source.append(source_info_address)

                if memory == 'minimum':
                    name_addr = channel_name_address
                    unit_addr = channel_unit_address
                    comment_addr = channel_comment_address
                else:
                    name_addr = 0
                    unit_addr = 0
                    comment_addr = 0

                s_type, s_size = fmt_to_datatype_v4(samples.dtype)

                # add channel block
                kargs = {
                    'channel_type': v4c.CHANNEL_TYPE_VALUE,
                    'bit_count': s_size,
                    'byte_offset': offset,
                    'bit_offset': 0,
                    'data_type': s_type,
                    'min_raw_value': 0,
                    'max_raw_value': 0,
                    'lower_limit': 0,
                    'upper_limit': 0,
                    'flags': 0,
                    'name_addr': name_addr,
                    'unit_addr': unit_addr,
                    'comment_addr': comment_addr,
                }
                ch = Channel(**kargs)
                if memory != 'minimum':
                    ch.name = name
                    ch.unit = signal.unit
                    ch.comment = signal.comment
                    gp_channels.append(ch)
                else:
                    address = tell()
                    write(bytes(ch))
                    gp_channels.append(address)

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
                    types.append((field_name, samples.dtype, shape))

                    gp_texts['conversion_tab'].append(None)

                    if memory == 'minimum':
                        block = TextBlock(text=name, meta=False)
                        channel_name_address = tell()
                        write(bytes(block))

                        if signal.unit:
                            block = TextBlock(text=signal.unit, meta=False)
                            channel_unit_address = tell()
                            write(bytes(block))
                        else:
                            channel_unit_address = 0

                        if signal.comment:
                            block = TextBlock(text=signal.comment, meta=False)
                            channel_comment_address = tell()
                            write(bytes(block))
                        else:
                            channel_comment_address = 0

                    # add channel conversion
                    if memory != 'minimum':
                        gp_conv.append(None)
                    else:
                        gp_conv.append(0)

                    # source for channel
                    if memory != 'minimum':
                        gp_source.append(source_block)
                    else:
                        gp_source.append(source_info_address)

                    if memory == 'minimum':
                        name_addr = channel_name_address
                        unit_addr = channel_unit_address
                        comment_addr = channel_comment_address
                    else:
                        name_addr = 0
                        unit_addr = 0
                        comment_addr = 0

                    # add channel dependency block
                    kargs = {
                        'dims': 1,
                        'ca_type': v4c.CA_TYPE_SCALE_AXIS,
                        'flags': 0,
                        'byte_offset_base': samples.dtype.itemsize,
                        'dim_size_0': shape[0],
                    }
                    dep = ChannelArrayBlock(**kargs)
                    gp_dep.append([dep, ])

                    # add components channel
                    min_val, max_val = get_min_max(samples)
                    s_type, s_size = fmt_to_datatype_v4(samples.dtype)
                    byte_size = max(s_size // 8, 1)
                    kargs = {
                        'channel_type': v4c.CHANNEL_TYPE_VALUE,
                        'bit_count': s_size,
                        'byte_offset': offset,
                        'bit_offset': 0,
                        'data_type': s_type,
                        'min_raw_value': min_val if min_val <= max_val else 0,
                        'max_raw_value': max_val if min_val <= max_val else 0,
                        'lower_limit': min_val if min_val <= max_val else 0,
                        'upper_limit': max_val if min_val <= max_val else 0,
                        'flags': v4c.FLAG_PHY_RANGE_OK | v4c.FLAG_VAL_RANGE_OK,
                        'name_addr': name_addr,
                        'unit_addr': unit_addr,
                        'comment_addr': comment_addr,
                    }

                    ch = Channel(**kargs)
                    if memory != 'minimum':
                        ch.name = name
                        ch.unit = signal.unit
                        ch.comment = signal.comment
                        gp_channels.append(ch)
                    else:

                        address = tell()
                        write(bytes(ch))
                        gp_channels.append(address)

                    parent_dep.referenced_channels.append((ch_cntr, dg_cntr))
                    for dim in shape:
                        byte_size *= dim
                    offset += byte_size

                    if name not in self.channels_db:
                        self.channels_db[name] = []
                    self.channels_db[name].append((dg_cntr, ch_cntr))

                    # update the parents as well
                    parents[ch_cntr] = field_name, 0

                    ch_cntr += 1

        # channel group
        kargs = {
            'cycles_nr': cycles_nr,
            'samples_byte_nr': offset,
        }
        gp['channel_group'] = ChannelGroup(**kargs)
        gp['size'] = cycles_nr * offset
        gp_texts['channel_group'].append(None)

        # data group
        gp['data_group'] = DataGroup()

        # data block
        if PYVERSION == 2:
            types = fix_dtype_fields(types)
        types = dtype(types)

        gp['sorted'] = True
        gp['types'] = types
        gp['parents'] = parents

        samples = fromarrays(fields, dtype=types)

        signals = None
        del signals

        try:
            block = samples.tostring()

            if memory == 'full':
                gp['data_location'] = v4c.LOCATION_MEMORY
                gp['data_block'] = DataBlock(data=block)
            else:
                gp['data_location'] = v4c.LOCATION_TEMPORARY_FILE

                data_address = self._tempfile.tell()
                gp['data_group']['data_block_addr'] = data_address
                self._tempfile.write(bytes(block))
        except MemoryError:
            if memory == 'full':
                raise
            else:
                gp['data_location'] = v4c.LOCATION_TEMPORARY_FILE

                data_address = self._tempfile.tell()
                gp['data_group']['data_block_addr'] = data_address
                for sample in samples:
                    self._tempfile.write(sample.tostring())

    def attach(self,
               data,
               file_name=None,
               comment=None,
               compression=True,
               mime=r'application/octet-stream'):
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
        text = """<FHcomment>
<TX>Added new embedded attachment from {}</TX>
<tool_id>asammdf</tool_id>
<tool_vendor>asammdf</tool_vendor>
<tool_version>{}</tool_version>
</FHcomment>"""
        text = text.format(
            file_name if file_name else 'bin.bin',
            __version__,
        )
        fh_text = TextBlock(text=text, meta=True)

        self.file_history.append((fh, fh_text))

        texts = {}
        texts['mime_addr'] = TextBlock(text=mime, meta=False)
        if comment:
            texts['comment_addr'] = TextBlock(text=comment, meta=False)
        text = file_name if file_name else 'bin.bin'
        texts['file_name_addr'] = TextBlock(text=text)
        at_block = AttachmentBlock(data=data, compression=compression)
        at_block['creator_index'] = creator_index
        self.attachments.append((at_block, texts))

    def close(self):
        """ if the MDF was created with memory=False and new
        channels have been appended, then this must be called just before the
        object is not used anymore to clean-up the temporary file"""
        if self._tempfile is not None:
            self._tempfile.close()
        if self._file is not None:
            self._file.close()

    def extract_attachment(self, index):
        """ extract attachment *index* data. If it is an embedded attachment,
        then this method creates the new file according to the attachment file
        name information

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

                file_path = (
                    texts['file_name_addr']['text']
                    .decode('utf-8')
                    .strip(' \n\t\0')
                )
                out_path = os.path.dirname(file_path)
                if out_path:
                    if not os.path.exists(out_path):
                        os.makedirs(out_path)

                with open(file_path, 'wb') as f:
                    f.write(data)

                return data
            else:
                # for external attachments read the file and return the content
                if flags & v4c.FLAG_AT_MD5_VALID:
                    file_path = (
                        texts['file_name_addr']['text']
                        .decode('utf-8')
                        .strip(' \n\t\0')
                    )
                    data = open(file_path, 'rb').read()
                    md5_worker = md5()
                    md5_worker.update(data)
                    md5_sum = md5_worker.digest()
                    if attachment['md5_sum'] == md5_sum:
                        if (texts['mime_addr']['text']
                                .decode('utf-8')
                                .startswith('text')):
                            with open(file_path, 'r') as f:
                                data = f.read()
                        return data
                    else:
                        message = (
                            'ATBLOCK md5sum="{}" '
                            'and external attachment data ({}) '
                            'md5sum="{}"'
                        )
                        message = message.format(
                            attachment['md5_sum'],
                            file_path,
                            md5_sum,
                        )
                        warnings.warn(message)
                else:
                    if (texts['mime_addr']['text']
                            .decode('utf-8')
                            .startswith('text')):
                        mode = 'r'
                    else:
                        mode = 'rb'
                    with open(file_path, mode) as f:
                        data = f.read()
                    return data
        except Exception as err:
            os.chdir(current_path)
            message = 'Exception during attachment extraction: ' + repr(err)
            warnings.warn(message)
            return b''

    def get_channel_unit(self, name=None, group=None, index=None):
        """Gets channel unit.

        Channel can be specified in two ways:

        * using the first positional argument *name*

            * if there are multiple occurrences for this channel then the
                *group* and *index* arguments can be used to select a specific
                group.
            * if there are multiple occurrences for this channel and either the
                *group* or *index* arguments is None then a warning is issued

        * using the group number (keyword argument *group*) and the channel
            number (keyword argument *index*). Use *info* method for group and
            channel numbers


        If the *raster* keyword argument is not *None* the output is
        interpolated accordingly.

        Parameters
        ----------
        name : string
            name of channel
        group : int
            0-based group index
        index : int
            0-based channel index

        Returns
        -------
        unit : str
            found channel unit

        """
        gp_nr, ch_nr = self._validate_channel_selection(
            name,
            group,
            index,
        )

        grp = self.groups[gp_nr]

        if grp['data_location'] == v4c.LOCATION_ORIGINAL_FILE:
            stream = self._file
        else:
            stream = self._tempfile

        channel = grp['channels'][ch_nr]
        conversion = grp['channel_conversions'][ch_nr]

        if self.memory == 'minimum':

            channel = Channel(
                address=channel,
                stream=stream,
            )

            if conversion:
                conversion = ChannelConversion(
                    address=conversion,
                    stream=stream,
                )

            address = (
                conversion and conversion['unit_addr']
                or channel['unit_addr']
                or 0
            )

            if address:
                unit = TextBlock(
                    address=address,
                    stream=stream,
                )
                if PYVERSION == 3:
                    unit = unit['text'].decode('utf-8').strip(' \n\t\0')
                else:
                    unit = unit['text'].strip(' \n\t\0')
            else:
                unit = ''
        else:
            unit = (
                conversion and conversion.unit
                or channel.unit
                or ''
            )

        return unit

    def get_channel_comment(self, name=None, group=None, index=None):
        """Gets channel comment.

        Channel can be specified in two ways:

        * using the first positional argument *name*

            * if there are multiple occurrences for this channel then the
                *group* and *index* arguments can be used to select a specific
                group.
            * if there are multiple occurrences for this channel and either the
                *group* or *index* arguments is None then a warning is issued

        * using the group number (keyword argument *group*) and the channel
            number (keyword argument *index*). Use *info* method for group and
            channel numbers


        If the *raster* keyword argument is not *None* the output is
        interpolated accordingly.

        Parameters
        ----------
        name : string
            name of channel
        group : int
            0-based group index
        index : int
            0-based channel index

        Returns
        -------
        comment : str
            found channel comment

        """
        gp_nr, ch_nr = self._validate_channel_selection(
            name,
            group,
            index,
        )

        grp = self.groups[gp_nr]

        if grp['data_location'] == v4c.LOCATION_ORIGINAL_FILE:
            stream = self._file
        else:
            stream = self._tempfile

        channel = grp['channels'][ch_nr]

        if self.memory == 'minimum':
            channel = Channel(
                address=channel,
                stream=stream,
            )

            address = channel['comment_addr']
            if address:
                comment_block = TextBlock(
                    address=address,
                    stream=stream,
                )
                comment = (
                    comment_block['text']
                    .decode('utf-8')
                    .strip(' \r\n\t\0')
                )
                if comment_block['id'] == b'##MD':
                    match = TX.search(comment)
                    if match:
                        comment = match.group('text')
                    else:
                        comment = ''
            else:
                comment = ''
        else:
            comment = channel.comment
            if channel.comment_type == b'##MD':
                match = TX.search(comment)
                if match:
                    comment = match.group('text')
                else:
                    comment = ''

        return comment

    def get(self,
            name=None,
            group=None,
            index=None,
            raster=None,
            samples_only=False,
            data=None,
            raw=False):
        """Gets channel samples.
        Channel can be specified in two ways:

        * using the first positional argument *name*

            * if there are multiple occurances for this channel then the
                *group* and *index* arguments can be used to select a specific
                group.
            * if there are multiple occurances for this channel and either the
                *group* or *index* arguments is None then a warning is issued

        * using the group number (keyword argument *group*) and the channel
            number (keyword argument *index*). Use *info* method for group and
            channel numbers

        If the *raster* keyword argument is not *None* the output is
        interpolated accordingly

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
            if *True* return only the channel samples as numpy array; if
                *False* return a *Signal* object
        data : bytes
            prevent redundant data read by providing the raw data group samples
        raw : bool
            return channel samples without appling the conversion rule; default
            `False`

        Returns
        -------
        res : (numpy.array | Signal)
            returns *Signal* if *samples_only*=*False* (default option),
            otherwise returns numpy.array
            The *Signal* samples are:

                * numpy recarray for channels that have composition/channel
                    array address or for channel of type BYTEARRAY,
                    CANOPENDATE, CANOPENTIME
                * numpy array for all the rest

        Raises
        ------
        MdfException :

        * if the channel name is not found
        * if the group index is out of range
        * if the channel index is out of range

        Examples
        --------
        >>> from asammdf import MDF, Signal
        >>> import numpy as np
        >>> t = np.arange(5)
        >>> s = np.ones(5)
        >>> mdf = MDF(version='4.10')
        >>> for i in range(4):
        ...     sigs = [Signal(s*(i*10+j), t, name='Sig') for j in range(1, 4)]
        ...     mdf.append(sigs)
        ...
        >>> # first group and channel index of the specified channel name
        ...
        >>> mdf.get('Sig')
        UserWarning: Multiple occurances for channel "Sig". Using first occurance from data group 4. Provide both "group" and "index" arguments to select another data group
        <Signal Sig:
                samples=[ 1.  1.  1.  1.  1.]
                timestamps=[0 1 2 3 4]
                unit=""
                info=None
                comment="">
        >>> # first channel index in the specified group
        ...
        >>> mdf.get('Sig', 1)
        <Signal Sig:
                samples=[ 11.  11.  11.  11.  11.]
                timestamps=[0 1 2 3 4]
                unit=""
                info=None
                comment="">
        >>> # channel named Sig from group 1 channel index 2
        ...
        >>> mdf.get('Sig', 1, 2)
        <Signal Sig:
                samples=[ 12.  12.  12.  12.  12.]
                timestamps=[0 1 2 3 4]
                unit=""
                info=None
                comment="">
        >>> # channel index 1 or group 2
        ...
        >>> mdf.get(None, 2, 1)
        <Signal Sig:
                samples=[ 21.  21.  21.  21.  21.]
                timestamps=[0 1 2 3 4]
                unit=""
                info=None
                comment="">
        >>> mdf.get(group=2, index=1)
        <Signal Sig:
                samples=[ 21.  21.  21.  21.  21.]
                timestamps=[0 1 2 3 4]
                unit=""
                info=None
                comment="">

        """
        gp_nr, ch_nr = self._validate_channel_selection(
            name,
            group,
            index,
        )

        memory = self.memory
        grp = self.groups[gp_nr]
        if grp['data_location'] == v4c.LOCATION_ORIGINAL_FILE:
            stream = self._file
        else:
            stream = self._tempfile
        if memory == 'minimum':
            channel = Channel(
                address=grp['channels'][ch_nr],
                stream=stream,
            )
            addr = grp['channel_conversions'][ch_nr]
            if addr:
                conversion = ChannelConversion(
                    address=addr,
                    stream=stream,
                )
            else:
                conversion = None
            if name is None:
                name = TextBlock(
                    address=channel['name_addr'],
                    stream=stream,
                )
                name = (
                    name['text']
                    .decode('utf-8')
                    .strip(' \r\t\n\0')
                    .split('\\')[0]
                )
            channel.name = name
        else:
            channel = grp['channels'][ch_nr]
            conversion = grp['channel_conversions'][ch_nr]
            name = channel.name

        dependency_list = grp['channel_dependencies'][ch_nr]
        cycles_nr = grp['channel_group']['cycles_nr']

        # get data group record
        try:
            parents, dtypes = grp['parents'], grp['types']
        except KeyError:
            grp['parents'], grp['types'] = self._prepare_record(grp)
            parents, dtypes = grp['parents'], grp['types']

        # get group data
        if data is None:
            data = self._load_group_data(grp)

        info = None

        # get the channel signal data if available
        signal_data = self._load_signal_data(
            channel['data_block_addr']
        )

        # check if this is a channel array
        if dependency_list:
            arrays = []
            name = channel.name

            if all(
                    not isinstance(dep, ChannelArrayBlock)
                    for dep in dependency_list):
                # structure channel composition
                if memory == 'minimum':
                    names = []
                    # TODO : get exactly he group and chanenl
                    for address in dependency_list:
                        channel = Channel(
                            address=address,
                            stream=stream,
                        )

                        name_ = get_text_v4(channel['name_addr'], stream)
                        names.append(name_)
                else:
                    names = [ch.name for ch in dependency_list]
                arrays = [
                    self.get(name_, samples_only=True, raw=raw)
                    for name_ in names
                ]

                types = [
                    (name_, arr.dtype)
                    for name_, arr in zip(names, arrays)
                ]
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
                except KeyError:
                    parent, bit_offset = None, None

                if parent is not None:
                    if 'record' not in grp:
                        if dtypes.itemsize:
                            record = fromstring(data, dtype=dtypes)
                        else:
                            record = None

                        if self.memory == 'full':
                            grp['record'] = record
                    else:
                        record = grp['record']

                    record.setflags(write=False)

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
                        shape = (ca_block['dim_size_0'],)
                        arrays.append(vals)
                        dtype_pair = channel.name, vals.dtype, shape
                        types.append(dtype_pair)

                    elif ca_block['ca_type'] == v4c.CA_TYPE_LOOKUP:
                        shape = vals.shape[1:]
                        arrays.append(vals)
                        dtype_pair = channel.name, vals.dtype, shape
                        types.append(dtype_pair)

                        if ca_block['flags'] & v4c.FLAG_CA_FIXED_AXIS:
                            for i in range(dims_nr):
                                shape = (ca_block['dim_size_{}'.format(i)],)
                                axis = []
                                for j in range(shape[0]):
                                    key = 'axis_{}_value_{}'.format(i, j)
                                    axis.append(ca_block[key])
                                axis = array([axis for _ in range(cycles_nr)])
                                arrays.append(axis)
                                dtype_pair = (
                                    'axis_{}'.format(i),
                                    axis.dtype,
                                    shape,
                                )
                                types.append(dtype_pair)
                        else:
                            for i in range(dims_nr):
                                ch_nr, dg_nr = ca_block.referenced_channels[i]
                                if memory == 'minimum':
                                    channel = Channel(
                                        address=self.groups[dg_nr]['channels'][ch_nr],
                                        stream=stream,
                                    )
                                    axisname = get_text_v4(
                                        channel['name_addr'],
                                        stream,
                                    )
                                else:
                                    axisname = (
                                        self.groups[dg_nr]
                                        ['channels']
                                        [ch_nr]
                                        .name
                                    )
                                shape = (ca_block['dim_size_{}'.format(i)],)
                                axis_values = self.get(
                                    group=dg_nr,
                                    index=ch_nr,
                                    samples_only=True,
                                )
                                axis_values = axis_values[axisname]
                                arrays.append(axis_values)
                                dtype_pair = (
                                    axisname,
                                    axis_values.dtype,
                                    shape,
                                )
                                types.append(dtype_pair)

                    elif ca_block['ca_type'] == v4c.CA_TYPE_ARRAY:
                        shape = vals.shape[1:]
                        arrays.append(vals)
                        dtype_pair = channel.name, vals.dtype, shape
                        types.append(dtype_pair)

                for ca_block in dependency_list[1:]:
                    dims_nr = ca_block['dims']

                    if ca_block['flags'] & v4c.FLAG_CA_FIXED_AXIS:
                        for i in range(dims_nr):
                            shape = (ca_block['dim_size_{}'.format(i)],)
                            axis = []
                            for j in range(shape[0]):
                                key = 'axis_{}_value_{}'.format(i, j)
                                axis.append(ca_block[key])
                            axis = array([axis for _ in range(cycles_nr)])
                            arrays.append(axis)
                            dtype_pair = 'axis_{}'.format(i), axis.dtype, shape
                            types.append(dtype_pair)
                    else:
                        for i in range(dims_nr):
                            ch_nr, dg_nr = ca_block.referenced_channels[i]
                            if memory == 'minimum':
                                channel = Channel(
                                    address=self.groups[dg_nr]['channels'][ch_nr],
                                    stream=stream,
                                )
                                axisname = get_text_v4(
                                    channel['name_addr'],
                                    stream,
                                )
                            else:
                                axisname = (
                                    self.groups[dg_nr]
                                    ['channels']
                                    [ch_nr]
                                    .name
                                )
                            shape = (ca_block['dim_size_{}'.format(i)],)
                            axis_values = self.get(
                                group=dg_nr,
                                index=ch_nr,
                                samples_only=True,
                            )
                            axis_values = axis_values[axisname]
                            arrays.append(axis_values)
                            dtype_pair = axisname, axis_values.dtype, shape
                            types.append(dtype_pair)

                if PYVERSION == 2:
                    types = fix_dtype_fields(types)

                vals = fromarrays(arrays, dtype(types))
        else:
            # get channel values
            if channel['channel_type'] in (v4c.CHANNEL_TYPE_VIRTUAL,
                                           v4c.CHANNEL_TYPE_VIRTUAL_MASTER):
                data_type = channel['data_type']
                ch_dtype = dtype(get_fmt_v4(data_type, 8))

                vals = arange(cycles_nr, dtype=ch_dtype)
                record = None
            else:
                try:
                    parent, bit_offset = parents[ch_nr]
                except KeyError:
                    parent, bit_offset = None, None

                if parent is not None:
                    if 'record' not in grp:
                        if dtypes.itemsize:
                            record = fromstring(data, dtype=dtypes)
                        else:
                            record = None

                        if memory == 'full':
                            grp['record'] = record
                    else:
                        record = grp['record']

                    record.setflags(write=False)

                    vals = record[parent]
                    bits = channel['bit_count']
                    size = vals.dtype.itemsize
                    data_type = channel['data_type']

                    if vals.dtype.kind not in 'ui' and (bit_offset or not bits == size * 8):
                        vals = self._get_not_byte_aligned_data(
                            data,
                            grp,
                            ch_nr,
                        )
                    else:
                        if bit_offset:
                            dtype_ = vals.dtype
                            if dtype_.kind == 'i':
                                vals = vals.astype(dtype('<u{}'.format(size)))
                                vals >>= bit_offset
                            else:
                                vals = vals >> bit_offset

                        if not bits == size * 8:
                            mask = (1 << bits) - 1
                            if vals.flags.writeable:
                                vals &= mask
                            else:
                                vals = vals & mask

                            if data_type in v4c.SIGNED_INT:
                                vals = as_non_byte_sized_signed_int(vals, bits)

                else:
                    vals = self._get_not_byte_aligned_data(data, grp, ch_nr)

            if cycles_nr:
                if conversion is None:
                    conversion_type = v4c.CONVERSION_TYPE_NON
                else:
                    conversion_type = conversion['conversion_type']

                if raw:
                    pass
                elif conversion_type == v4c.CONVERSION_TYPE_NON:
                    # check if it is VLDS channel type with SDBLOCK
                    data_type = channel['data_type']
                    channel_type = channel['channel_type']

                    if channel_type in (v4c.CHANNEL_TYPE_VALUE,
                                        v4c.CHANNEL_TYPE_MLSD):
                        if v4c.DATA_TYPE_STRING_LATIN_1 \
                                <= data_type \
                                <= v4c.DATA_TYPE_STRING_UTF_16_BE:
                            vals = [val.tobytes() for val in vals]

                            if data_type == v4c.DATA_TYPE_STRING_UTF_16_BE:
                                encoding = 'utf-16-be'

                            elif data_type == v4c.DATA_TYPE_STRING_UTF_16_LE:
                                encoding = 'utf-16-le'

                            elif data_type == v4c.DATA_TYPE_STRING_UTF_8:
                                encoding = 'utf-8'

                            elif data_type == v4c.DATA_TYPE_STRING_LATIN_1:
                                encoding = 'latin-1'

                            vals = array(
                                [x.decode(encoding).strip('\0') for x in vals]
                            )
                            vals = encode(vals, 'latin-1')

                        # CANopen date
                        elif data_type == v4c.DATA_TYPE_CANOPEN_DATE:

                            vals = vals.tostring()

                            types = dtype(
                                [('ms', '<u2'),
                                 ('min', '<u1'),
                                 ('hour', '<u1'),
                                 ('day', '<u1'),
                                 ('month', '<u1'),
                                 ('year', '<u1')]
                            )
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

                            names = [
                                'ms',
                                'min',
                                'hour',
                                'day',
                                'month',
                                'year',
                                'summer_time',
                                'day_of_week',
                            ]
                            vals = fromarrays(arrays, names=names)

                        # CANopen time
                        elif data_type == v4c.DATA_TYPE_CANOPEN_TIME:
                            vals = vals.tostring()

                            types = dtype(
                                [('ms', '<u4'),
                                 ('days', '<u2')]
                            )
                            dates = fromstring(vals, types)

                            arrays = []
                            # bits 28 to 31 are reserverd for ms
                            arrays.append(dates['ms'] & 0xFFFFFFF)
                            arrays.append(dates['days'] & 0x3F)

                            names = ['ms', 'days']
                            vals = fromarrays(arrays, names=names)

                        # byte array
                        elif data_type == v4c.DATA_TYPE_BYTEARRAY:
                            vals = vals.tostring()
                            size = max(bits >> 3, 1)

                            vals = frombuffer(
                                vals,
                                dtype=dtype('({},)u1'.format(size)),
                            )

                            types = [(channel.name, vals.dtype, vals.shape[1:])]
                            if PYVERSION == 2:
                                types = fix_dtype_fields(types)

                            types = dtype(types)
                            arrays = [vals, ]

                            vals = fromarrays(arrays, dtype=types)

                    elif channel_type == v4c.CHANNEL_TYPE_VLSD:
                        if signal_data:
                            values = []
                            for offset in vals:
                                offset = int(offset)
                                str_size = unpack_from('<I', signal_data, offset)[0]
                                values.append(
                                    signal_data[offset + 4: offset + 4 + str_size]
                                )

                            if data_type == v4c.DATA_TYPE_STRING_UTF_16_BE:
                                vals = [v.decode('utf-16-be') for v in values]

                            elif data_type == v4c.DATA_TYPE_STRING_UTF_16_LE:
                                vals = [v.decode('utf-16-le') for v in values]

                            elif data_type == v4c.DATA_TYPE_STRING_UTF_8:
                                vals = [v.decode('utf-8') for v in values]

                            elif data_type == v4c.DATA_TYPE_STRING_LATIN_1:
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
                    if (P1, P2, P3, P4, P5, P6) != (0, 1, 0, 0, 0, 1):
                        X = vals
                        vals = evaluate(v4c.CONV_RAT_TEXT)

                elif conversion_type == v4c.CONVERSION_TYPE_ALG:
                    if not memory == 'minimum':
                        formula = conversion.formula
                    else:
                        block = TextBlock(
                            address=conversion['formula_addr'],
                            stream=stream,
                        )
                        formula = (
                            block['text']
                            .decode('utf-8')
                            .strip(' \n\t\0')
                        )
                    X = vals
                    vals = evaluate(formula)

                elif conversion_type in (v4c.CONVERSION_TYPE_TABI,
                                         v4c.CONVERSION_TYPE_TAB):
                    nr = conversion['val_param_nr'] // 2
                    raw_vals = array(
                        [conversion['raw_{}'.format(i)] for i in range(nr)]
                    )
                    phys = array(
                        [conversion['phys_{}'.format(i)] for i in range(nr)]
                    )
                    if conversion_type == v4c.CONVERSION_TYPE_TABI:
                        vals = interp(vals, raw_vals, phys)
                    else:
                        idx = searchsorted(raw_vals, vals)
                        idx = clip(idx, 0, len(raw_vals) - 1)
                        vals = phys[idx]

                elif conversion_type == v4c.CONVERSION_TYPE_RTAB:
                    nr = (conversion['val_param_nr'] - 1) // 3
                    lower = array(
                        [conversion['lower_{}'.format(i)] for i in range(nr)]
                    )
                    upper = array(
                        [conversion['upper_{}'.format(i)] for i in range(nr)]
                    )
                    phys = array(
                        [conversion['phys_{}'.format(i)] for i in range(nr)]
                    )
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
                        size = max(bits >> 3, 1)
                        ch_fmt = get_fmt_v4(channel['data_type'], size)
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
                        size = max(bits >> 3, 1)
                        ch_fmt = get_fmt_v4(channel['data_type'], size)
                        vals = array(res).astype(ch_fmt)

                elif conversion_type == v4c.CONVERSION_TYPE_TABX:
                    nr = conversion['val_param_nr']
                    raw_vals = array(
                        [conversion['val_{}'.format(i)] for i in range(nr)]
                    )

                    if not memory == 'minimum':
                        phys = array(
                            [grp['texts']['conversion_tab'][ch_nr]['text_{}'.format(i)]['text']
                             for i in range(nr)]
                        )
                        default = grp['texts']['conversion_tab'][ch_nr] \
                            .get('default_addr', {}) \
                            .get('text', b'')
                    else:
                        phys = []
                        for i in range(nr):
                            address = (
                                grp['texts']
                                ['conversion_tab']
                                [ch_nr]
                                ['text_{}'.format(i)]
                            )
                            if address:
                                block = TextBlock(
                                    address=address,
                                    stream=stream,
                                )
                                phys.append(block['text'])
                            else:
                                phys.append(b'')
                        phys = array(phys)

                        if grp['texts']['conversion_tab'][ch_nr].get(
                                'default_addr',
                                0):
                            block = TextBlock(
                                address=grp['texts']['conversion_tab'][ch_nr]['default_addr'],
                                stream=stream,
                            )
                            default = block['text']
                        else:
                            default = b''
                    info = {
                        'raw': raw_vals,
                        'phys': phys,
                        'default': default,
                    }

                elif conversion_type == v4c.CONVERSION_TYPE_RTABX:
                    nr = conversion['val_param_nr'] // 2

                    if not memory == 'minimum':
                        phys = array(
                            [grp['texts']['conversion_tab'][ch_nr]['text_{}'.format(i)]['text']
                             for i in range(nr)]
                        )
                        default = grp['texts']['conversion_tab'][ch_nr] \
                            .get('default_addr', {}) \
                            .get('text', b'')
                    else:
                        phys = []
                        for i in range(nr):
                            address = grp['texts']['conversion_tab'][ch_nr]['text_{}'.format(i)]
                            if address:
                                block = TextBlock(
                                    address=address,
                                    stream=stream,
                                )
                                phys.append(block['text'])
                            else:
                                phys.append(b'')
                        phys = array(phys)
                        if grp['texts']['conversion_tab'][ch_nr].get(
                                'default_addr',
                                0):
                            block = TextBlock(
                                address=grp['texts']['conversion_tab'][ch_nr]['default_addr'],
                                stream=stream,
                            )
                            default = block['text']
                        else:
                            default = b''
                    lower = array(
                        [conversion['lower_{}'.format(i)] for i in range(nr)]
                    )
                    upper = array(
                        [conversion['upper_{}'.format(i)] for i in range(nr)]
                    )

                    info = {
                        'lower': lower,
                        'upper': upper,
                        'phys': phys,
                        'default': default,
                    }

                elif conversion_type == v4c.CONVERSION_TYPE_TTAB:
                    nr = conversion['val_param_nr'] - 1

                    if memory == 'minimum':
                        raw_vals = []
                        for i in range(nr):
                            block = TextBlock(
                                address=grp['texts']['conversion_tab'][ch_nr]['text_{}'.format(i)],
                                stream=stream,
                            )
                            raw_vals.append(block['text'])
                        raw_vals = array(raw_vals)
                    else:
                        raw_vals = array(
                            [grp['texts']['conversion_tab'][ch_nr]['text_{}'.format(i)]['text']
                             for i in range(nr)]
                        )
                    phys = array(
                        [conversion['val_{}'.format(i)] for i in range(nr)]
                    )
                    default = conversion['val_default']
                    info = {
                        'raw': raw_vals,
                        'phys': phys,
                        'default': default,
                    }

                elif conversion_type == v4c.CONVERSION_TYPE_TRANS:
                    nr = (conversion['ref_param_nr'] - 1) // 2
                    if memory == 'minimum':
                        in_ = []
                        for i in range(nr):
                            block = TextBlock(
                                address=grp['texts']['conversion_tab'][ch_nr]['input_{}_addr'.format(i)],
                                stream=stream,
                            )
                            in_.append(block['text'])
                        in_ = array(in_)

                        out_ = []
                        for i in range(nr):
                            block = TextBlock(
                                address=grp['texts']['conversion_tab'][ch_nr]['output_{}_addr'.format(i)],
                                stream=stream,
                            )
                            out_.append(block['text'])
                        out_ = array(out_)

                        block = TextBlock(
                            address=grp['texts']['conversion_tab'][ch_nr]['default_addr'],
                            stream=stream,
                        )
                        default = block['text']
                    else:
                        in_ = array(
                            [grp['texts']['conversion_tab'][ch_nr]['input_{}_addr'.format(i)]['text']
                             for i in range(nr)]
                        )
                        out_ = array(
                            [grp['texts']['conversion_tab'][ch_nr]['output_{}_addr'.format(i)]['text']
                             for i in range(nr)]
                        )
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
                    info = {
                        'input': in_,
                        'output': out_,
                        'default': default,
                    }

        # in case of invalidation bits, valid_index will hold the valid indexes
        valid_index = None
        if grp['channel_group']['invalidation_bytes_nr']:

            if channel['flags'] & (
                    v4c.FLAG_INVALIDATION_BIT_VALID | v4c.FLAG_ALL_SAMPLES_VALID) == v4c.FLAG_INVALIDATION_BIT_VALID:
                ch_invalidation_pos = channel['pos_invalidation_bit']
                pos_byte, pos_offset = divmod(ch_invalidation_pos, 8)
                mask = 1 << pos_offset

                if record is None:
                    record = fromstring(data, dtype=dtypes)
                    record.setflags(write=False)

                inval_bytes = record['invalidation_bytes']
                inval_index = array(
                    [bytes_[pos_byte] & mask for bytes_ in inval_bytes]
                )
                valid_index = argwhere(inval_index == 0).flatten()
                vals = vals[valid_index]

        if samples_only:
            res = vals
        else:
            # search for unit in conversion texts

            if memory == 'minimum':

                address = (
                    conversion and conversion['unit_addr']
                    or channel['unit_addr']
                    or 0
                )
                if address:
                    unit = get_text_v4(
                        address=address,
                        stream=stream,
                    )
                else:
                    unit = ''

                address = channel['comment_addr']
                if address:
                    comment = get_text_v4(
                        address=address,
                        stream=stream,
                    )

                    if channel.comment_type == b'##MD':
                        match = TX.search(comment)
                        if match:
                            comment = match.group('text')
                        else:
                            comment = ''
                else:
                    comment = ''
            else:
                unit = (
                    conversion and conversion.unit
                    or channel.unit
                    or ''
                )
                comment = channel.comment
                if channel.comment_type == b'##MD':
                    match = TX.search(comment)
                    if match:
                        comment = match.group('text')

            t = self.get_master(gp_nr, data)

            # consider invalidation bits
            if valid_index is not None:
                t = t[valid_index]

            res = Signal(
                samples=vals,
                timestamps=t,
                unit=unit,
                name=name,
                comment=comment,
                info=info,
            )

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
        except KeyError:
            pass

        group = self.groups[index]

        if group['data_location'] == v4c.LOCATION_ORIGINAL_FILE:
            stream = self._file
        else:
            stream = self._tempfile
        memory = self.memory

        time_ch_nr = self.masters_db.get(index, None)
        cycles_nr = group['channel_group']['cycles_nr']

        if time_ch_nr is None:
            t = arange(cycles_nr, dtype=float64)
        else:
            time_conv = group['channel_conversions'][time_ch_nr]
            if memory == 'minimum':
                if time_conv:
                    time_conv = ChannelConversion(
                        address=group['channel_conversions'][time_ch_nr],
                        stream=stream,
                    )
                else:
                    time_conv = None
                time_ch = Channel(
                    address=group['channels'][time_ch_nr],
                    stream=stream,
                )
            else:
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
                    parents, dtypes = self._prepare_record(group)
                    group['parents'], group['types'] = parents, dtypes

                # get data
                if data is None:
                    data = self._load_group_data(group)

                try:
                    parent, _ = parents[time_ch_nr]
                except KeyError:
                    parent = None
                if parent is not None:
                    not_found = object()
                    record = group.get('record', not_found)
                    if record is not_found:
                        if dtypes.itemsize:
                            record = fromstring(data, dtype=dtypes)
                        else:
                            record = None

                        if memory == 'full':
                            group['record'] = record

                    record.setflags(write=False)
                    t = record[parent]
                else:
                    t = self._get_not_byte_aligned_data(
                        data, group,
                        time_ch_nr,
                    )

                # get timestamps
                if time_conv:
                    if time_conv['conversion_type'] == v4c.CONVERSION_TYPE_LIN:
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
        info['version'] = self.identification['version_str'] \
            .decode('utf-8') \
            .strip(' \n\t\0')
        info['groups'] = len(self.groups)
        for i, gp in enumerate(self.groups):
            if gp['data_location'] == v4c.LOCATION_ORIGINAL_FILE:
                stream = self._file
            elif gp['data_location'] == v4c.LOCATION_TEMPORARY_FILE:
                stream = self._tempfile
            inf = {}
            info['group {}'.format(i)] = inf
            inf['cycles'] = gp['channel_group']['cycles_nr']
            inf['channels count'] = len(gp['channels'])
            for j, channel in enumerate(gp['channels']):
                if self.memory == 'minimum':
                    channel = Channel(
                        address=channel,
                        stream=stream,
                    )
                    name = TextBlock(
                        address=channel['name_addr'],
                        stream=stream,
                    )
                    name = (
                        name['text']
                        .decode('utf-8')
                        .strip(' \r\t\n\0')
                        .split('\\')[0]
                    )
                else:
                    name = channel.name

                ch_type = v4c.CHANNEL_TYPE_TO_DESCRIPTION[channel['channel_type']]
                inf['channel {}'.format(j)] = 'name="{}" type={}'.format(
                    name,
                    ch_type,
                )

        return info

    def save(self, dst='', overwrite=None, compression=0):
        """Save MDF to *dst*. If *dst* is not provided the the destination file
        name is the MDF name. If overwrite is *True* then the destination file
        is overwritten, otherwise the file name is appened with '_<cntr>', were
        '<cntr>' is the first conter that produces a new file name
        (that does not already exist in the filesystem)

        Parameters
        ----------
        dst : str
            destination file name, Default ''
        overwrite : bool
            overwrite flag, default *False*
        compression : int
            use compressed data blocks, default 0; valid since version 4.10

            * 0 - no compression
            * 1 - deflate (slower, but produces smaller files)
            * 2 - transposition + deflate (slowest, but produces
                the smallest files)

        Returns
        -------
        output_file : str
            output file name

        """
        if overwrite is None:
            overwrite = self._overwrite
        output_file = ''

        if self.name is None and dst == '':
            message = (
                'Must specify a destination file name '
                'for MDF created from scratch'
            )
            raise MdfException(message)
        else:
            if self.memory == 'minimum':
                output_file = self._save_without_metadata(
                    dst,
                    overwrite,
                    compression,
                )
            else:
                output_file = self._save_with_metadata(
                    dst,
                    overwrite,
                    compression,
                )
        return output_file

    def _save_with_metadata(self, dst, overwrite, compression):
        """Save MDF to *dst*. If *dst* is not provided the the destination file
        name is the MDF name. If overwrite is *True* then the destination file
        is overwritten, otherwise the file name is appened with '_<cntr>', were
        '<cntr>' is the first conter that produces a new file name
        (that does not already exist in the filesystem)

        Parameters
        ----------
        dst : str
            destination file name, Default ''
        overwrite : bool
            overwrite flag, default *False*
        compression : int
            use compressed data blocks, default 0; valid since version 4.10

            * 0 - no compression
            * 1 - deflate (slower, but produces smaller files)
            * 2 - transposition + deflate (slowest, but produces
                the smallest files)

        """
        if self.name is None and dst == '':
            message = ('Must specify a destination file name '
                       'for MDF created from scratch')
            raise MdfException(message)

        dst = dst if dst else self.name
        if not dst.endswith(('mf4', 'MF4')):
            dst = dst + '.mf4'
        if overwrite is False:
            if os.path.isfile(dst):
                cntr = 0
                while True:
                    name = os.path.splitext(dst)[0] + '_{}.mf4'.format(cntr)
                    if not os.path.isfile(name):
                        break
                    else:
                        cntr += 1
                message = (
                    'Destination file "{}" already exists '
                    'and "overwrite" is False. Saving MDF file as "{}"'
                )
                message = message.format(dst, name)
                warnings.warn(message)
                dst = name

        if not self.file_history:
            comment = 'created'
        else:
            comment = 'updated'

        text = """<FHcomment>
<TX>{}</TX>
<tool_id>asammdf</tool_id>
<tool_vendor>asammdf</tool_vendor>
<tool_version>{}</tool_version>
</FHcomment>"""
        text = text.format(comment, __version__)
        self.file_history.append(
            [FileHistory(),
             TextBlock(text=text, meta=True)]
        )

        if self.memory == 'low' and dst == self.name:
            destination = dst + '.temp'
        else:
            destination = dst

        with open(destination, 'wb+') as dst_:
            defined_texts = {}

            write = dst_.write
            tell = dst_.tell
            seek = dst_.seek

            write(bytes(self.identification))
            write(bytes(self.header))

            original_data_addresses = []

            if compression == 1:
                zip_type = v4c.FLAG_DZ_DEFLATE
            else:
                zip_type = v4c.FLAG_DZ_TRANPOSED_DEFLATE

            # write DataBlocks first
            for gp in self.groups:

                original_data_addresses.append(
                    gp['data_group']['data_block_addr']
                )
                address = tell()

                data = self._load_group_data(gp)

                if MDF4._split_data_blocks:
                    samples_size = gp['channel_group']['samples_byte_nr']
                    split_size = MDF4._split_threshold // samples_size
                    split_size *= samples_size
                    if split_size == 0:
                        chunks = 1
                    else:
                        chunks = len(data) / split_size
                        chunks = int(ceil(chunks))
                else:
                    chunks = 1

                if chunks == 1:
                    if compression and self.version != '4.00':
                        if compression == 1:
                            param = 0
                        else:
                            param = gp['channel_group']['samples_byte_nr']
                        kargs = {
                            'data': data,
                            'zip_type': zip_type,
                            'param': param,
                        }
                        data_block = DataZippedBlock(**kargs)
                    else:
                        data_block = DataBlock(data=data)
                    write(bytes(data_block))

                    align = data_block['block_len'] % 8
                    if align:
                        write(b'\0' * (8 - align))

                    if gp['channel_group']['cycles_nr']:
                        gp['data_group']['data_block_addr'] = address
                    else:
                        gp['data_group']['data_block_addr'] = 0
                else:
                    kargs = {
                        'flags': v4c.FLAG_DL_EQUAL_LENGHT,
                        'zip_type': zip_type,
                    }
                    hl_block = HeaderList(**kargs)

                    kargs = {
                        'flags': v4c.FLAG_DL_EQUAL_LENGHT,
                        'links_nr': chunks + 1,
                        'data_block_nr': chunks,
                        'data_block_len': split_size,
                    }
                    dl_block = DataList(**kargs)

                    data_blocks = []
                    for i in range(chunks):
                        data_ = data[i * split_size: (i + 1) * split_size]
                        if compression and self.version != '4.00':
                            if compression == 1:
                                zip_type = v4c.FLAG_DZ_DEFLATE
                            else:
                                zip_type = v4c.FLAG_DZ_TRANPOSED_DEFLATE
                            if compression == 1:
                                param = 0
                            else:
                                param = gp['channel_group']['samples_byte_nr']
                            kargs = {
                                'data': data_,
                                'zip_type': zip_type,
                                'param': param,
                            }
                            block = DataZippedBlock(**kargs)
                        else:
                            block = DataBlock(data=data_)
                        address = tell()
                        block.address = address
                        data_blocks.append(block)

                        write(bytes(block))

                        align = block['block_len'] % 8
                        if align:
                            write(b'\0' * (8 - align))
                        dl_block['data_block_addr{}'.format(i)] = address

                    address = tell()
                    dl_block.address = address
                    write(bytes(dl_block))

                    if compression and self.version != '4.00':
                        hl_block['first_dl_addr'] = address
                        address = tell()
                        hl_block.address = address
                        write(bytes(hl_block))

                    gp['data_group']['data_block_addr'] = address

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
                        blocks.append(b'\0' * (8 - align))
                        address += at_block['block_len'] + 8 - align
                    else:
                        address += at_block['block_len']

                for i, (at_block, text) in enumerate(self.attachments[:-1]):
                    at_block['next_at_addr'] = self.attachments[i + 1][0].address
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
                fh['next_fh_addr'] = self.file_history[i + 1][0].address
            self.file_history[-1][0]['next_fh_addr'] = 0

            # data groups
            gp_rec_ids = []
            for gp in self.groups:
                gp_rec_ids.append(gp['data_group']['record_id_len'])
                gp['data_group']['record_id_len'] = 0
                gp['data_group'].address = address
                address += gp['data_group']['block_len']
                blocks.append(gp['data_group'])

                gp['data_group']['comment_addr'] = 0

            for i, dg in enumerate(self.groups[:-1]):
                addr_ = self.groups[i + 1]['data_group'].address
                dg['data_group']['next_dg_addr'] = addr_
            self.groups[-1]['data_group']['next_dg_addr'] = 0

            tab_conversion = (
                v4c.CONVERSION_TYPE_TABX,
                v4c.CONVERSION_TYPE_RTABX,
                v4c.CONVERSION_TYPE_TTAB,
                v4c.CONVERSION_TYPE_TRANS,
            )

            si_map = {}

            # go through each data group and append the rest of the blocks
            for i, gp in enumerate(self.groups):
                # write TXBLOCK's
                for item_list in gp['texts'].values():
                    for dict_ in item_list:
                        if dict_ is None:
                            continue
                        for key, tx_block in dict_.items():
                            # text blocks can be shared
                            text = tx_block['text']
                            if text in defined_texts:
                                tx_block.address = defined_texts[text]
                            else:
                                defined_texts[text] = address
                                tx_block.address = address
                                address += tx_block['block_len']
                                blocks.append(tx_block)

                for channel in gp['channels']:
                    if channel.name:
                        tx_block = TextBlock(text=channel.name)
                        text = tx_block['text']
                        if text in defined_texts:
                            channel['name_addr'] = defined_texts[text]
                        else:
                            channel['name_addr'] = address
                            defined_texts[text] = address
                            tx_block.address = address
                            address += tx_block['block_len']
                            blocks.append(tx_block)
                    else:
                        channel['name_addr'] = 0

                    if channel.unit:
                        tx_block = TextBlock(text=channel.unit)
                        text = tx_block['text']
                        if text in defined_texts:
                            channel['unit_addr'] = defined_texts[text]
                        else:
                            channel['unit_addr'] = address
                            defined_texts[text] = address
                            tx_block.address = address
                            address += tx_block['block_len']
                            blocks.append(tx_block)
                    else:
                        channel['unit_addr'] = 0

                    if channel.comment:
                        meta = channel.comment_type == b'##MD'
                        tx_block = TextBlock(text=channel.comment, meta=meta)
                        text = tx_block['text']
                        if text in defined_texts:
                            channel['comment_addr'] = defined_texts[text]
                        else:
                            channel['comment_addr'] = address
                            defined_texts[text] = address
                            tx_block.address = address
                            address += tx_block['block_len']
                            blocks.append(tx_block)
                    else:
                        channel['comment_addr'] = 0

                for source in gp['channel_sources']:
                    if source:
                        if source.name:
                            tx_block = TextBlock(text=source.name)
                            text = tx_block['text']
                            if text in defined_texts:
                                source['name_addr'] = defined_texts[text]
                            else:
                                source['name_addr'] = address
                                defined_texts[text] = address
                                tx_block.address = address
                                address += tx_block['block_len']
                                blocks.append(tx_block)
                        else:
                            source['name_addr'] = 0

                        if source.path:
                            tx_block = TextBlock(text=source.path)
                            text = tx_block['text']
                            if text in defined_texts:
                                source['path_addr'] = defined_texts[text]
                            else:
                                source['path_addr'] = address
                                defined_texts[text] = address
                                tx_block.address = address
                                address += tx_block['block_len']
                                blocks.append(tx_block)
                        else:
                            source['path_addr'] = 0

                        if source.comment:
                            tx_block = TextBlock(text=source.comment)
                            text = tx_block['text']
                            if text in defined_texts:
                                source['comment_addr'] = defined_texts[text]
                            else:
                                source['comment_addr'] = address
                                defined_texts[text] = address
                                tx_block.address = address
                                address += tx_block['block_len']
                                blocks.append(tx_block)
                        else:
                            source['comment_addr'] = 0

                for conversion in gp['channel_conversions']:
                    if conversion:
                        if conversion.name:
                            tx_block = TextBlock(text=conversion.name)
                            text = tx_block['text']
                            if text in defined_texts:
                                conversion['name_addr'] = defined_texts[text]
                            else:
                                conversion['name_addr'] = address
                                defined_texts[text] = address
                                tx_block.address = address
                                address += tx_block['block_len']
                                blocks.append(tx_block)
                        else:
                            conversion['name_addr'] = 0

                        if conversion.unit:
                            tx_block = TextBlock(text=conversion.unit)
                            text = tx_block['text']
                            if text in defined_texts:
                                conversion['unit_addr'] = defined_texts[text]
                            else:
                                conversion['unit_addr'] = address
                                defined_texts[text] = address
                                tx_block.address = address
                                address += tx_block['block_len']
                                blocks.append(tx_block)
                        else:
                            conversion['unit_addr'] = 0

                        if conversion.comment:
                            tx_block = TextBlock(text=conversion.comment)
                            text = tx_block['text']
                            if text in defined_texts:
                                conversion['comment_addr'] = defined_texts[text]
                            else:
                                conversion['comment_addr'] = address
                                defined_texts[text] = address
                                tx_block.address = address
                                address += tx_block['block_len']
                                blocks.append(tx_block)
                        else:
                            conversion['comment_addr'] = 0

                        if conversion['conversion_type'] == v4c.CONVERSION_TYPE_ALG and conversion.formula:
                            tx_block = TextBlock(text=conversion.formula)
                            text = tx_block['text']
                            if text in defined_texts:
                                conversion['formula_addr'] = defined_texts[text]
                            else:
                                conversion['formula_addr'] = address
                                defined_texts[text] = address
                                tx_block.address = address
                                address += tx_block['block_len']
                                blocks.append(tx_block)

                # channel conversions
                for j, conv in enumerate(gp['channel_conversions']):
                    if conv:
                        conv.address = address

                        conv['inv_conv_addr'] = 0

                        if conv['conversion_type'] in tab_conversion:
                            for key in gp['texts']['conversion_tab'][j]:
                                conv[key] = (
                                    gp['texts']
                                    ['conversion_tab']
                                    [j]
                                    [key]
                                    .address
                                )

                        address += conv['block_len']
                        blocks.append(conv)

                # channel sources
                for j, source in enumerate(gp['channel_sources']):
                    if source:
                        source_id = id(source)
                        if source_id not in si_map:
                            source.address = address
                            address += source['block_len']
                            blocks.append(source)
                            si_map[source_id] = 0

                # channel data
                gp_sd = gp['signal_data'] = []
                for j, ch in enumerate(gp['channels']):
                    signal_data = self._load_signal_data(ch['data_block_addr'])
                    if signal_data:
                        signal_data = SignalDataBlock(data=signal_data)
                        signal_data.address = address
                        address += signal_data['block_len']
                        blocks.append(signal_data)
                        align = signal_data['block_len'] % 8
                        if align % 8:
                            blocks.append(b'\0' * (8 - align))
                            address += 8 - align
                        gp_sd.append(signal_data)
                    else:
                        gp_sd.append(None)

                # channel dependecies
                for j, dep_list in enumerate(gp['channel_dependencies']):
                    if dep_list:
                        if all(isinstance(dep, ChannelArrayBlock)
                               for dep in dep_list):
                            for dep in dep_list:
                                dep.address = address
                                address += dep['block_len']
                                blocks.append(dep)
                            for k, dep in enumerate(dep_list[:-1]):
                                dep['composition_addr'] = dep_list[k + 1].address
                            dep_list[-1]['composition_addr'] = 0

                # channels
                for j, (channel, signal_data) in enumerate(
                        zip(gp['channels'], gp['signal_data'])):
                    channel.address = address

                    address += channel['block_len']
                    blocks.append(channel)

                    if not gp['channel_conversions'][j]:
                        channel['conversion_addr'] = 0
                    else:
                        addr_ = gp['channel_conversions'][j].address
                        channel['conversion_addr'] = addr_
                    if gp['channel_sources'][j]:
                        addr_ = gp['channel_sources'][j].address
                        channel['source_addr'] = addr_
                    else:
                        channel['source_addr'] = 0
                    if signal_data:
                        channel['data_block_addr'] = signal_data.address
                    else:
                        channel['data_block_addr'] = 0

                    if gp['channel_dependencies'][j]:
                        addr_ = gp['channel_dependencies'][j][0].address
                        channel['component_addr'] = addr_

                group_channels = gp['channels']
                if group_channels:
                    for j, channel in enumerate(group_channels[:-1]):
                        channel['next_ch_addr'] = group_channels[j + 1].address
                    group_channels[-1]['next_ch_addr'] = 0

                # channel dependecies
                j = 0
                while j < len(gp['channels']):
                    dep_list = gp['channel_dependencies'][j]
                    if dep_list and all(
                            isinstance(dep, Channel) for dep in dep_list):
                        gp['channels'][j]['component_addr'] = dep_list[0].address
                        gp['channels'][j]['next_ch_addr'] = dep_list[-1]['next_ch_addr']
                        dep_list[-1]['next_ch_addr'] = 0
                        j += len(dep_list)

                        for dep in dep_list:
                            dep['source_addr'] = 0
                    else:
                        j += 1

                # channel group
                gp['channel_group'].address = address
                gp['channel_group']['first_ch_addr'] = gp['channels'][0].address
                gp['channel_group']['next_cg_addr'] = 0
                cg_texts = gp['texts']['channel_group'][0]
                for key in ('acq_name_addr', 'comment_addr'):
                    if cg_texts and key in cg_texts:
                        addr_ = gp['texts']['channel_group'][0][key].address
                        gp['channel_group'][key] = addr_
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

            for gp, rec_id in zip(self.groups, gp_rec_ids):
                gp['data_group']['record_id_len'] = rec_id

            if self.groups:
                addr_ = self.groups[0]['data_group'].address
                self.header['first_dg_addr'] = addr_
            else:
                self.header['first_dg_addr'] = 0
            self.header['file_history_addr'] = self.file_history[0][0].address
            if self.attachments:
                addr_ = self.attachments[0][0].address
                self.header['first_attachment_addr'] = addr_
            else:
                self.header['first_attachment_addr'] = 0
            if self.file_comment:
                self.header['comment_addr'] = self.file_comment.address
            else:
                self.header['comment_addr'] = 0

            seek(v4c.IDENTIFICATION_BLOCK_SIZE, v4c.SEEK_START)
            write(bytes(self.header))

            for orig_addr, gp in zip(original_data_addresses, self.groups):
                gp['data_group']['data_block_addr'] = orig_addr

        if self.memory == 'low' and dst == self.name:
            self.close()
            os.remove(self.name)
            os.rename(destination, self.name)

            self.groups = []
            self.header = None
            self.identification = None
            self.file_history = []
            self.channels_db = {}
            self.masters_db = {}
            self.attachments = []
            self.file_comment = None

            self._ch_map = {}
            self._master_channel_cache = {}

            self._tempfile = TemporaryFile()
            self._file = open(self.name, 'rb')
            self._read()

        return dst

    def _save_without_metadata(self, dst, overwrite, compression):
        """Save MDF to *dst*. If *dst* is not provided the the destination file
        name is the MDF name. If overwrite is *True* then the destination file
        is overwritten, otherwise the file name is appened with '_<cntr>', were
        '<cntr>' is the first conter that produces a new file name
        (that does not already exist in the filesystem)

        Parameters
        ----------
        dst : str
            destination file name, Default ''
        overwrite : bool
            overwrite flag, default *False*
        compression : int
            use compressed data blocks, default 0; valid since version 4.10

            * 0 - no compression
            * 1 - deflate (slower, but produces smaller files)
            * 2 - transposition + deflate (slowest, but produces
                the smallest files)

        """
        if self.name is None and dst == '':
            message = (
                'Must specify a destination file name '
                'for MDF created from scratch'
            )
            raise MdfException(message)

        dst = dst if dst else self.name
        if not dst.endswith(('mf4', 'MF4')):
            dst = dst + '.mf4'
        if overwrite is False:
            if os.path.isfile(dst):
                cntr = 0
                while True:
                    name = os.path.splitext(dst)[0] + '_{}.mf4'.format(cntr)
                    if not os.path.isfile(name):
                        break
                    else:
                        cntr += 1
                message = (
                    'Destination file "{}" already exists '
                    'and "overwrite" is False. Saving MDF file as "{}"'
                )
                message = message.format(dst, name)
                warnings.warn(message)
                dst = name

        if not self.file_history:
            comment = 'created'
        else:
            comment = 'updated'

        text = """<FHcomment>
<TX>{}</TX>
<tool_id>asammdf</tool_id>
<tool_vendor>asammdf</tool_vendor>
<tool_version>{}</tool_version>
</FHcomment>"""
        text = text.format(comment, __version__)
        self.file_history.append(
            [FileHistory(),
             TextBlock(text=text, meta=True)]
        )

        if dst == self.name:
            destination = dst + '.temp'
        else:
            destination = dst

        with open(destination, 'wb+') as dst_:
            defined_texts = {}

            write = dst_.write
            tell = dst_.tell
            seek = dst_.seek

            write(bytes(self.identification))
            write(bytes(self.header))

            original_data_addresses = []

            if compression == 1:
                zip_type = v4c.FLAG_DZ_DEFLATE
            else:
                zip_type = v4c.FLAG_DZ_TRANPOSED_DEFLATE

            # write DataBlocks first
            for gp in self.groups:

                original_data_addresses.append(
                    gp['data_group']['data_block_addr']
                )
                address = tell()

                data = self._load_group_data(gp)

                if MDF4._split_data_blocks:
                    samples_size = gp['channel_group']['samples_byte_nr']
                    split_size = MDF4._split_threshold // samples_size
                    split_size *= samples_size
                    if split_size == 0:
                        chunks = 1
                    else:
                        chunks = len(data) / split_size
                        chunks = int(ceil(chunks))
                else:
                    chunks = 1

                if chunks == 1:
                    if compression and self.version != '4.00':
                        if compression == 1:
                            param = 0
                        else:
                            param = gp['channel_group']['samples_byte_nr']
                        kargs = {
                            'data': data,
                            'zip_type': zip_type,
                            'param': param,
                        }
                        data_block = DataZippedBlock(**kargs)
                    else:
                        data_block = DataBlock(data=data)
                    write(bytes(data_block))

                    align = data_block['block_len'] % 8
                    if align:
                        write(b'\0' * (8 - align))

                    if gp['channel_group']['cycles_nr']:
                        gp['data_group']['data_block_addr'] = address
                    else:
                        gp['data_group']['data_block_addr'] = 0
                else:
                    kargs = {
                        'flags': v4c.FLAG_DL_EQUAL_LENGHT,
                        'zip_type': zip_type,
                    }
                    hl_block = HeaderList(**kargs)

                    kargs = {
                        'flags': v4c.FLAG_DL_EQUAL_LENGHT,
                        'links_nr': chunks + 1,
                        'data_block_nr': chunks,
                        'data_block_len': split_size,
                    }
                    dl_block = DataList(**kargs)

                    data_blocks = []
                    for i in range(chunks):
                        data_ = data[i * split_size: (i + 1) * split_size]
                        if compression and self.version != '4.00':
                            if compression == 1:
                                zip_type = v4c.FLAG_DZ_DEFLATE
                            else:
                                zip_type = v4c.FLAG_DZ_TRANPOSED_DEFLATE
                            if compression == 1:
                                param = 0
                            else:
                                param = gp['channel_group']['samples_byte_nr']
                            kargs = {
                                'data': data_,
                                'zip_type': zip_type,
                                'param': param,
                            }
                            block = DataZippedBlock(**kargs)
                        else:
                            block = DataBlock(data=data_)
                        address = tell()
                        block.address = address
                        data_blocks.append(block)

                        write(bytes(block))

                        align = block['block_len'] % 8
                        if align:
                            write(b'\0' * (8 - align))
                        dl_block['data_block_addr{}'.format(i)] = address

                    address = tell()
                    dl_block.address = address
                    write(bytes(dl_block))

                    if compression and self.version != '4.00':
                        hl_block['first_dl_addr'] = address
                        address = tell()
                        hl_block.address = address
                        write(bytes(hl_block))

                    gp['data_group']['data_block_addr'] = address

            address = tell()

            if self.file_comment:
                address = tell()
                self.file_comment.address = address
                write(bytes(self.file_comment))

            # attachemnts
            address = tell()
            blocks = []
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
                        blocks.append(b'\0' * (8 - align))
                        address += at_block['block_len'] + 8 - align
                    else:
                        address += at_block['block_len']

                for i, (at_block, text) in enumerate(self.attachments[:-1]):
                    at_block['next_at_addr'] = self.attachments[i + 1][0].address
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
                fh['next_fh_addr'] = self.file_history[i + 1][0].address
            self.file_history[-1][0]['next_fh_addr'] = 0

            for blk in blocks:
                write(bytes(blk))

            blocks = []

            tab_conversion = (
                v4c.CONVERSION_TYPE_TABX,
                v4c.CONVERSION_TYPE_RTABX,
                v4c.CONVERSION_TYPE_TTAB,
                v4c.CONVERSION_TYPE_TRANS,
            )

            si_map = {}

            # go through each data group and append the rest of the blocks
            for i, gp in enumerate(self.groups):
                gp['temp_channels'] = ch_addrs = []
                gp['temp_channel_conversions'] = cc_addrs = []
                gp['temp_channel_sources'] = si_addrs = []

                if gp['data_location'] == v4c.LOCATION_ORIGINAL_FILE:
                    stream = self._file
                else:
                    stream = self._tempfile

                temp_texts = deepcopy(gp['texts'])
                # write TXBLOCK's
                for item_list in temp_texts.values():
                    for dict_ in item_list:
                        if not dict_:
                            continue
                        for key, tx_block in dict_.items():
                            # text blocks can be shared
                            block = TextBlock(
                                address=tx_block,
                                stream=stream,
                            )
                            text = block['text']
                            if text in defined_texts:
                                dict_[key] = defined_texts[text]
                            else:
                                address = tell()
                                defined_texts[text] = address
                                dict_[key] = address
                                write(bytes(block))

                for source in gp['channel_sources']:
                    if source:
                        stream.seek(source, v4c.SEEK_START)
                        raw_bytes = stream.read(v4c.SI_BLOCK_SIZE)
                        if raw_bytes in si_map:
                            si_addrs.append(si_map[raw_bytes])
                        else:
                            source = SourceInformation(
                                raw_bytes=raw_bytes,
                            )

                            if source['name_addr']:
                                tx_block = TextBlock(
                                    address=source['name_addr'],
                                    stream=stream,
                                )
                                text = tx_block['text']
                                if text in defined_texts:
                                    source['name_addr'] = defined_texts[text]
                                else:
                                    address = tell()
                                    source['name_addr'] = address
                                    defined_texts[text] = address
                                    tx_block.address = address
                                    write(bytes(tx_block))
                            else:
                                source['name_addr'] = 0

                            if source.path:
                                tx_block = TextBlock(
                                    address=source['path_addr'],
                                    stream=stream,
                                )
                                text = tx_block['text']
                                if text in defined_texts:
                                    source['path_addr'] = defined_texts[text]
                                else:
                                    address = tell()
                                    source['path_addr'] = address
                                    defined_texts[text] = address
                                    tx_block.address = address
                                    write(bytes(tx_block))
                            else:
                                source['path_addr'] = 0

                            if source['comment_addr']:
                                tx_block = TextBlock(
                                    address=source['comment_addr'],
                                    stream=stream,
                                )
                                text = tx_block['text']
                                if text in defined_texts:
                                    source['comment_addr'] = defined_texts[text]
                                else:
                                    address = tell()
                                    source['comment_addr'] = address
                                    defined_texts[text] = address
                                    tx_block.address = address
                                    write(bytes(tx_block))
                            else:
                                source['comment_addr'] = 0

                            address = tell()
                            si_addrs.append(address)
                            si_map[raw_bytes] = address
                            write(bytes(source))
                    else:
                        si_addrs.append(0)

                for j, conversion in enumerate(gp['channel_conversions']):
                    if conversion:
                        conversion = ChannelConversion(
                            address=conversion,
                            stream=stream,
                        )

                        if conversion['name_addr']:
                            tx_block = TextBlock(
                                address=conversion['name_addr'],
                                stream=stream,
                            )
                            text = tx_block['text']
                            if text in defined_texts:
                                conversion['name_addr'] = defined_texts[text]
                            else:
                                address = tell()
                                conversion['name_addr'] = address
                                defined_texts[text] = address
                                tx_block.address = address
                                write(bytes(tx_block))
                        else:
                            conversion['name_addr'] = 0

                        if conversion['unit_addr']:
                            tx_block = TextBlock(
                                address=conversion['unit_addr'],
                                stream=stream,
                            )
                            text = tx_block['text']
                            if text in defined_texts:
                                conversion['unit_addr'] = defined_texts[text]
                            else:
                                address = tell()
                                conversion['unit_addr'] = address
                                defined_texts[text] = address
                                tx_block.address = address
                                write(bytes(tx_block))
                        else:
                            conversion['unit_addr'] = 0

                        if conversion['comment_addr']:
                            tx_block = TextBlock(
                                address=conversion['comment_addr'],
                                stream=stream,
                            )
                            text = tx_block['text']
                            if text in defined_texts:
                                conversion['comment_addr'] = defined_texts[text]
                            else:
                                address = tell()
                                conversion['comment_addr'] = address
                                defined_texts[text] = address
                                tx_block.address = address
                                write(bytes(tx_block))
                        else:
                            conversion['comment_addr'] = 0

                        if conversion['conversion_type'] == v4c.CONVERSION_TYPE_ALG and conversion['formula_addr']:
                            tx_block = TextBlock(
                                address=conversion['formula_addr'],
                                stream=stream,
                            )
                            text = tx_block['text']
                            if text in defined_texts:
                                conversion['formula_addr'] = defined_texts[text]
                            else:
                                address = tell()
                                conversion['formula_addr'] = address
                                defined_texts[text] = address
                                tx_block.address = address
                                write(bytes(tx_block))

                        elif conversion['conversion_type'] in tab_conversion:
                            for key in temp_texts['conversion_tab'][j]:
                                conversion[key] = temp_texts['conversion_tab'][j][key]

                        conversion['inv_conv_addr'] = 0

                        address = tell()
                        cc_addrs.append(address)
                        write(bytes(conversion))
                    else:
                        cc_addrs.append(0)

                # channel dependecies
                temp_deps = []
                for j, dep_list in enumerate(gp['channel_dependencies']):
                    if dep_list:
                        if all(isinstance(dep, ChannelArrayBlock)
                               for dep in dep_list):
                            temp_deps.append([])

                            for dep in dep_list:
                                address = tell()
                                dep.address = address
                                temp_deps[-1].append(address)
                                write(bytes(dep))
                            for k, dep in enumerate(dep_list[:-1]):
                                dep['composition_addr'] = dep_list[k + 1].address
                            dep_list[-1]['composition_addr'] = 0
                        else:
                            temp_deps.append([])
                            for _ in dep_list:
                                temp_deps[-1].append(0)
                    else:
                        temp_deps.append(0)

                # channels
                blocks = []
                chans = []
                address = blocks_start_addr = tell()

                gp['channel_group']['first_ch_addr'] = address


                for j, channel in enumerate(gp['channels']):
                    channel = Channel(
                        address=channel,
                        stream=stream,
                    )
                    channel.address = address
                    ch_addrs.append(address)
                    chans.append(channel)
                    blocks.append(channel)

                    address += channel['block_len']

                    if channel['name_addr']:
                        tx_block = TextBlock(
                            address=channel['name_addr'],
                            stream=stream,
                        )
                        text = tx_block['text']
                        if text in defined_texts:
                            channel['name_addr'] = defined_texts[text]
                        else:
                            channel['name_addr'] = address
                            defined_texts[text] = address
                            tx_block.address = address
                            address += tx_block['block_len']
                            blocks.append(tx_block)
                    else:
                        channel['name_addr'] = 0

                    if channel['unit_addr']:
                        tx_block = TextBlock(
                            address=channel['unit_addr'],
                            stream=stream,
                        )
                        text = tx_block['text']
                        if text in defined_texts:
                            channel['unit_addr'] = defined_texts[text]
                        else:
                            channel['unit_addr'] = address
                            defined_texts[text] = address
                            tx_block.address = address
                            address += tx_block['block_len']
                            blocks.append(tx_block)
                    else:
                        channel['unit_addr'] = 0

                    if channel['comment_addr']:
                        tx_block = TextBlock(
                            address=channel['comment_addr'],
                            stream=stream,
                        )
                        text = tx_block['text']
                        if text in defined_texts:
                            channel['comment_addr'] = defined_texts[text]
                        else:
                            channel['comment_addr'] = address
                            defined_texts[text] = address
                            tx_block.address = address
                            address += tx_block['block_len']
                            blocks.append(tx_block)
                    else:
                        channel['comment_addr'] = 0

                    channel['conversion_addr'] = gp['temp_channel_conversions'][j]
                    channel['source_addr'] = gp['temp_channel_sources'][j]
                    signal_data = self._load_signal_data(channel['data_block_addr'])
                    if signal_data:
                        signal_data = SignalDataBlock(data=signal_data)
                        blocks.append(signal_data)
                        channel['data_block_addr'] = address
                        address += signal_data['block_len']
                        align = signal_data['block_len'] % 8
                        if align % 8:
                            blocks.append(b'\0' * (8 - align))
                            address += 8 - align
                    else:
                        channel['data_block_addr'] = 0

                    if gp['channel_dependencies'][j]:
                        block = gp['channel_dependencies'][j][0]
                        if isinstance(block, (ChannelArrayBlock, Channel)):
                            channel['component_addr'] = block.address
                        else:
                            channel['component_addr'] = block

                group_channels = gp['channels']
                if group_channels:
                    for j, channel in enumerate(chans[:-1]):
                        channel['next_ch_addr'] = chans[j + 1].address
                    chans[-1]['next_ch_addr'] = 0

                # channel dependecies
                j = 0
                while j < len(gp['channels']):
                    dep_list = gp['channel_dependencies'][j]
                    if dep_list and all(
                            not isinstance(dep, ChannelArrayBlock) for dep in dep_list):

                        dep = chans[j+1]

                        channel = chans[j]
                        channel['component_addr'] = dep.address

                        dep = chans[j+len(dep_list)]
                        channel['next_ch_addr'] = dep['next_ch_addr']
                        dep['next_ch_addr'] = 0

                        for k, _ in enumerate(dep_list):
                            dep = chans[j+1+k]
                            dep['source_addr'] = 0

                        j += len(dep_list)
                    else:
                        j += 1

                seek(blocks_start_addr, v4c.SEEK_START)

                for block in blocks:
                    write(bytes(block))

                blocks = []
                chans = []
                address = tell()

                # channel group
                gp['channel_group'].address = address
                gp['channel_group']['next_cg_addr'] = 0
                cg_texts = temp_texts['channel_group'][0]
                for key in ('acq_name_addr', 'comment_addr'):
                    if cg_texts and key in cg_texts:
                        addr_ = temp_texts['channel_group'][0][key]
                        gp['channel_group'][key] = addr_
                gp['channel_group']['acq_source_addr'] = 0

                gp['data_group']['first_cg_addr'] = address

                write(bytes(gp['channel_group']))
                address = tell()

                del gp['temp_channel_sources']
                del gp['temp_channel_conversions']

            temp_texts = None

            blocks = []
            address = tell()
            gp_rec_ids = []
            # data groups
            for gp in self.groups:
                gp['data_group'].address = address
                gp_rec_ids.append(gp['data_group']['record_id_len'])
                gp['data_group']['record_id_len'] = 0
                address += gp['data_group']['block_len']
                blocks.append(gp['data_group'])

                gp['data_group']['comment_addr'] = 0

            for i, dg in enumerate(self.groups[:-1]):
                addr_ = self.groups[i + 1]['data_group'].address
                dg['data_group']['next_dg_addr'] = addr_
            self.groups[-1]['data_group']['next_dg_addr'] = 0

            for block in blocks:
                write(bytes(block))

            for gp, rec_id in zip(self.groups, gp_rec_ids):
                gp['data_group']['record_id_len'] = rec_id

            if self.groups:
                addr_ = self.groups[0]['data_group'].address
                self.header['first_dg_addr'] = addr_
            else:
                self.header['first_dg_addr'] = 0
            self.header['file_history_addr'] = self.file_history[0][0].address
            if self.attachments:
                addr_ = self.attachments[0][0].address
                self.header['first_attachment_addr'] = addr_
            else:
                self.header['first_attachment_addr'] = 0
            if self.file_comment:
                self.header['comment_addr'] = self.file_comment.address
            else:
                self.header['comment_addr'] = 0

            seek(v4c.IDENTIFICATION_BLOCK_SIZE, v4c.SEEK_START)
            write(bytes(self.header))

            for orig_addr, gp in zip(original_data_addresses, self.groups):
                gp['data_group']['data_block_addr'] = orig_addr

            for gp in self.groups:
                for dep_list in gp['channel_dependencies']:
                    if dep_list:
                        if all(
                                isinstance(dep, ChannelArrayBlock)
                                for dep in dep_list):
                            for dep in dep_list:
                                for i, (ch_nr, gp_nr) in enumerate(dep.referenced_channels):
                                    grp = self.groups[gp_nr]
                                    stream.seek(0, v4c.SEEK_END)

                                    dep['scale_axis_{}_dg_addr'.format(i)] = grp['data_group'].address
                                    dep['scale_axis_{}_cg_addr'.format(i)] = grp['channel_group'].address
                                    dep['scale_axis_{}_ch_addr'.format(i)] = grp['temp_channels'][ch_nr]
                                seek(dep.address, v4c.SEEK_START)
                                write(bytes(dep))

            for gp in self.groups:
                del gp['temp_channels']

        if dst == self.name:
            self.close()
            os.remove(self.name)
            os.rename(destination, self.name)

            self.groups = []
            self.header = None
            self.identification = None
            self.file_history = []
            self.channels_db = {}
            self.masters_db = {}
            self.attachments = []
            self.file_comment = None

            self._ch_map = {}
            self._master_channel_cache = {}

            self._tempfile = TemporaryFile()
            self._file = open(self.name, 'rb')
            self._read()
        return dst

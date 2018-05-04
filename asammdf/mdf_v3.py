# -*- coding: utf-8 -*-
""" ASAM MDF version 3 file format module """

from __future__ import division, print_function

import logging
import os
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from copy import deepcopy
from functools import reduce
from itertools import product
from math import ceil
from struct import unpack
from tempfile import TemporaryFile

from numpy import (
    arange,
    array,
    array_equal,
    column_stack,
    concatenate,
    dtype,
    flip,
    float64,
    interp,
    packbits,
    roll,
    uint8,
    union1d,
    unpackbits,
    zeros,
)

from numpy.core.records import fromarrays, fromstring

from .signal import Signal
from . import v2_v3_constants as v23c
from .conversion_utils import conversion_transfer
from .utils import (
    CHANNEL_COUNT,
    CONVERT_LOW,
    CONVERT_MINIMUM,
    MdfException,
    SignalSource,
    as_non_byte_sized_signed_int,
    fix_dtype_fields,
    fmt_to_datatype_v3,
    get_fmt_v3,
    get_min_max,
    get_unique_name,
    get_text_v3,
    validate_memory_argument,
    validate_version_argument,
    count_channel_groups,
)
from .v2_v3_blocks import (
    Channel,
    ChannelConversion,
    ChannelDependency,
    ChannelExtension,
    ChannelGroup,
    DataBlock,
    DataGroup,
    FileIdentificationBlock,
    HeaderBlock,
    ProgramBlock,
    TextBlock,
    TriggerBlock,
)
from .version import __version__

PYVERSION = sys.version_info[0]
if PYVERSION == 2:
    # pylint: disable=W0622
    from .utils import bytes
    # pylint: enable=W0622

logger = logging.getLogger('asammdf')

__all__ = ['MDF3', ]


def write_cc(conversion, defined_texts, blocks=None, address=None, stream=None):
    if conversion:
        if stream:
            tell = stream.tell
            write = stream.write

        for key, item in conversion.referenced_blocks.items():
            if isinstance(item, TextBlock):
                text = item['text']
                if text in defined_texts:
                    conversion[key] = defined_texts[text]
                else:
                    if stream:
                        address = tell()
                    conversion[key] = address
                    defined_texts[text] = address
                    item.address = address
                    if stream:
                        write(bytes(item))
                    else:
                        address += item['block_len']
                        blocks.append(item)

            elif isinstance(item, ChannelConversion):

                if stream:
                    write_cc(item, defined_texts, blocks, stream=stream)
                    address = tell()
                    item.address = address
                    conversion[key] = address
                    write(bytes(item))
                else:
                    item.address = address
                    conversion[key] = address
                    address += item['block_len']
                    blocks.append(item)
                    address = write_cc(item, defined_texts, blocks, address)

    return address


class MDF3(object):
    """If the *name* exist it will be loaded otherwise an empty file will be
    created that can be later saved to disk

    Parameters
    ----------
    name : string
        mdf file name
    memory : str
        memory optimization option; default `full`

        * if *full* the data group binary data block will be memorised in RAM
        * if *low* the channel data is read from disk on request, and the
          metadata is memorised into RAM
        * if *minimum* only minimal data is memorised into RAM

    version : string
        mdf file version ('2.00', '2.10', '2.14', '3.00', '3.10', '3.20' or
        '3.30'); default '3.30'

    Attributes
    ----------
    channels_db : dict
        used for fast channel access by name; for each name key the value is a
        list of (group index, channel index) tuples
    file_history : TextBlock
        file history text block; can be None
    groups : list
        list of data groups
    header : HeaderBlock
        mdf file header
    identification : FileIdentificationBlock
        mdf file start block
    masters_db : dict
        used for fast master channel access; for each group index key the value
         is the master channel index
    memory : str
        memory optimization option
    name : string
        mdf file name
    version : str
        mdf version

    """

    _terminate = False

    def __init__(self, name=None, memory='full', version='3.30', callback=None):
        memory = validate_memory_argument(memory)
        self.groups = []
        self.header = None
        self.identification = None
        self.file_history = None
        self.name = name
        self.memory = memory
        self.channels_db = {}
        self.masters_db = {}
        self.version = version

        self._master_channel_cache = {}
        self._master_channel_metadata = {}

        # used for appending to MDF created with memory=False
        self._tempfile = TemporaryFile()
        self._tempfile.write(b'\0')
        self._file = None

        self._read_fragment_size = 0
        self._write_fragment_size = 8 * 2 ** 20
        self._use_display_names = False
        self._single_bit_uint_as_bool = False

        self._callback = callback

        if name:
            self._file = open(self.name, 'rb')
            self._read()
        else:
            version = validate_version_argument(version, hint=3)
            self.identification = FileIdentificationBlock(version=version)
            self.version = version
            self.header = HeaderBlock(version=self.version)

    def _load_group_data(self, group):
        """ get group's data block bytes"""

        offset = 0
        if self.memory == 'full':
            yield group['data_block']['data'], offset
        else:
            channel_group = group['channel_group']

            if group['data_location'] == v23c.LOCATION_ORIGINAL_FILE:
                # go to the first data block of the current data group
                stream = self._file
            else:
                stream = self._tempfile

            # go to the first data block of the current data group
            if group['sorted']:
                samples_size = channel_group['samples_byte_nr']
                if self._read_fragment_size:
                    split_size = self._read_fragment_size // samples_size
                    split_size *= samples_size
                else:
                    channels_nr = len(group['channels'])

                    if self.memory == 'minimum':
                        y_axis = CONVERT_MINIMUM
                    else:
                        y_axis = CONVERT_LOW
                    split_size = interp(
                        channels_nr,
                        CHANNEL_COUNT,
                        y_axis,
                    )

                    split_size = int(split_size)

                    split_size = split_size // samples_size
                    split_size *= samples_size

                if split_size == 0:
                    split_size = samples_size

                blocks = zip(
                    group['data_block_addr'],
                    group['data_block_size'],
                )
                if PYVERSION == 2:
                    blocks = iter(blocks)

                cur_size = 0
                data = []

                while True:
                    try:
                        address, size = next(blocks)
                        current_address = address
                    except StopIteration:
                        break
                    stream.seek(address)

                    while size >= split_size - cur_size:
                        stream.seek(current_address)
                        if data:
                            data.append(stream.read(split_size - cur_size))
                            yield b''.join(data), offset
                            current_address += split_size - cur_size
                        else:
                            yield stream.read(split_size), offset
                            current_address += split_size
                        offset += split_size

                        size -= split_size - cur_size
                        data = []
                        cur_size = 0

                    if size:
                        stream.seek(current_address)
                        data.append(stream.read(size))
                        cur_size += size
                        offset += size

                if data:
                    yield b''.join(data), offset
                elif not offset:
                    yield b'', 0

            else:
                record_id = group['channel_group']['record_id']
                if PYVERSION == 2:
                    record_id = chr(record_id)
                cg_size = group['record_size']
                if group['data_group']['record_id_nr'] <= 2:
                    record_id_nr = group['data_group']['record_id_nr']
                else:
                    record_id_nr = 0
                cg_data = []

                blocks = zip(
                    group['data_block_addr'],
                    group['data_block_size'],
                )
                if PYVERSION == 2:
                    blocks = iter(blocks)

                for address, size in blocks:

                    stream.seek(address)
                    data = stream.read(size)

                    i = 0
                    while i < size:
                        rec_id = data[i]
                        # skip record id
                        i += 1
                        rec_size = cg_size[rec_id]
                        if rec_id == record_id:
                            rec_data = data[i: i + rec_size]
                            cg_data.append(rec_data)
                        # consider the second record ID if it exists
                        if record_id_nr == 2:
                            i += rec_size + 1
                        else:
                            i += rec_size
                    cg_data = b''.join(cg_data)
                    size = len(cg_data)
                    yield cg_data, offset
                    offset += size

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

        memory = self.memory
        if group['data_location'] == v23c.LOCATION_ORIGINAL_FILE:
            stream = self._file
        else:
            stream = self._tempfile
        grp = group
        record_size = grp['channel_group']['samples_byte_nr'] << 3
        next_byte_aligned_position = 0
        types = []
        current_parent = ""
        parent_start_offset = 0
        parents = {}
        group_channels = set()

        if memory != 'minimum':
            channels = grp['channels']
        else:
            channels = [
                Channel(address=ch_addr, stream=stream)
                for ch_addr in grp['channels']
            ]

        # the channels are first sorted ascending (see __lt__ method of Channel
        # class): a channel with lower start offset is smaller, when two
        # channels havethe same start offset the one with higer bit size is
        # considered smaller. The reason is that when the numpy record is built
        # and there are overlapping channels, the parent fields mustbe bigger
        # (bit size) than the embedded channels. For each channel the parent
        # dict will have a (parent name, bit offset) pair: the channel value is
        # computed using the values from the parent field, and the bit offset,
        # which is the channel's bit offset within the parent bytes.
        # This means all parents will have themselves as parent, and bit offset
        # of 0. Gaps in the records are also considered. Non standard integers
        # size is adjusted to the first higher standard integer size (eq. uint
        # of 28bits will be adjusted to 32bits)

        sortedchannels = sorted(enumerate(channels), key=lambda i: i[1])
        for original_index, new_ch in sortedchannels:
            # skip channels with channel dependencies from the numpy record
            if new_ch['ch_depend_addr']:
                continue

            start_offset = new_ch['start_offset']
            try:
                additional_byte_offset = new_ch['aditional_byte_offset']
                start_offset += 8 * additional_byte_offset
            except KeyError:
                pass

            bit_offset = start_offset % 8
            data_type = new_ch['data_type']
            bit_count = new_ch['bit_count']
            if memory == 'minimum':
                if new_ch.get('long_name_addr', 0):
                    name = get_text_v3(new_ch['long_name_addr'], stream)
                else:
                    name = (
                        new_ch['short_name']
                        .decode('latin-1')
                        .strip(' \r\n\t\0')
                    )
            else:
                name = new_ch.name

            # handle multiple occurance of same channel name
            name = get_unique_name(group_channels, name)
            group_channels.add(name)

            if start_offset >= next_byte_aligned_position:
                parent_start_offset = (start_offset // 8) * 8

                # check if there are byte gaps in the record
                gap = (parent_start_offset - next_byte_aligned_position) // 8
                if gap:
                    types.append(('', 'a{}'.format(gap)))

                # adjust size to 1, 2, 4 or 8 bytes for nonstandard integers
                size = bit_offset + bit_count
                if data_type == v23c.DATA_TYPE_STRING:
                    next_byte_aligned_position = parent_start_offset + size
                    if next_byte_aligned_position <= record_size:
                        dtype_pair = (name, get_fmt_v3(data_type, size))
                        types.append(dtype_pair)
                        parents[original_index] = name, bit_offset
                    else:
                        next_byte_aligned_position = parent_start_offset

                elif data_type == v23c.DATA_TYPE_BYTEARRAY:
                    next_byte_aligned_position = parent_start_offset + size
                    if next_byte_aligned_position <= record_size:
                        dtype_pair = (name, get_fmt_v3(data_type, size))
                        types.append(dtype_pair)
                        parents[original_index] = name, bit_offset
                    else:
                        next_byte_aligned_position = parent_start_offset

                else:
                    if size > 32:
                        next_byte_aligned_position = parent_start_offset + 64
                    elif size > 16:
                        next_byte_aligned_position = parent_start_offset + 32
                    elif size > 8:
                        next_byte_aligned_position = parent_start_offset + 16
                    else:
                        next_byte_aligned_position = parent_start_offset + 8

                    if next_byte_aligned_position <= record_size:
                        dtype_pair = (name, get_fmt_v3(data_type, size))
                        types.append(dtype_pair)
                        parents[original_index] = name, bit_offset
                    else:
                        next_byte_aligned_position = parent_start_offset

                current_parent = name
            else:
                max_overlapping = next_byte_aligned_position - start_offset
                if max_overlapping >= bit_count:
                    parents[original_index] = (
                        current_parent,
                        start_offset - parent_start_offset,
                    )
            if next_byte_aligned_position > record_size:
                break

        gap = (record_size - next_byte_aligned_position) >> 3
        if gap:
            dtype_pair = ('', 'a{}'.format(gap))
            types.append(dtype_pair)

        if PYVERSION == 2:
            types = fix_dtype_fields(types)

        return parents, dtype(types)

    def _get_not_byte_aligned_data(self, data, group, ch_nr):

        big_endian_types = (
            v23c.DATA_TYPE_UNSIGNED_MOTOROLA,
            v23c.DATA_TYPE_FLOAT_MOTOROLA,
            v23c.DATA_TYPE_DOUBLE_MOTOROLA,
            v23c.DATA_TYPE_SIGNED_MOTOROLA,
        )

        record_size = group['channel_group']['samples_byte_nr']

        if self.memory != 'minimum':
            channel = group['channels'][ch_nr]
        else:
            if group['data_location'] == v23c.LOCATION_ORIGINAL_FILE:
                channel = Channel(
                    address=group['channels'][ch_nr],
                    stream=self._file,
                )
            else:
                channel = Channel(
                    address=group['channels'][ch_nr],
                    stream=self._tempfile,
                )

        bit_offset = channel['start_offset'] % 8
        byte_offset = channel['start_offset'] // 8
        bit_count = channel['bit_count']

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

        fmt = get_fmt_v3(channel['data_type'], bit_count)
        if size <= byte_count:
            if channel['data_type'] in big_endian_types:
                types = [
                    ('', 'a{}'.format(byte_count -  size)),
                    ('vals', fmt),
                ]
            else:
                types = [
                    ('vals', fmt),
                    ('', 'a{}'.format(byte_count -  size)),
                ]
        else:
            types = [('vals', fmt), ]

        vals = fromstring(vals, dtype=dtype(types))['vals']

        if channel['data_type'] in v23c.SIGNED_INT:
            return as_non_byte_sized_signed_int(vals, bit_count)
        else:
            return vals

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
                        logger.warning(message)
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

    def _read(self):
        stream = self._file
        memory = self.memory

        cg_count = count_channel_groups(stream, 3)
        if self._callback:
            self._callback(0, cg_count)
        current_cg_index = 0

        # performance optimization
        read = stream.read
        seek = stream.seek

        dg_cntr = 0
        seek(0)

        self.identification = FileIdentificationBlock(
            stream=stream,
        )
        self.header = HeaderBlock(stream=stream)

        self.version = (
            self.identification['version_str']
            .decode('latin-1')
            .strip(' \n\t\0')
        )

        self.file_history = TextBlock(
            address=self.header['comment_addr'],
            stream=stream,
        )

        # this will hold mapping from channel address to Channel object
        # needed for linking dependency blocks to referenced channels after
        # the file is loaded
        ch_map = {}
        ce_map = {}
        cc_map = {}

        # go to first date group
        dg_addr = self.header['first_dg_addr']
        # read each data group sequentially
        while dg_addr:
            data_group = DataGroup(
                address=dg_addr,
                stream=stream,
            )
            record_id_nr = data_group['record_id_nr']
            cg_nr = data_group['cg_nr']
            cg_addr = data_group['first_cg_addr']
            data_addr = data_group['data_block_addr']

            # read trigger information if available
            trigger_addr = data_group['trigger_addr']
            if trigger_addr:
                trigger = TriggerBlock(
                    address=trigger_addr,
                    stream=stream,
                )
                if trigger['text_addr']:
                    trigger_text = TextBlock(
                        address=trigger['text_addr'],
                        stream=stream,
                    )
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
                grp['texts'] = {
                    'conversion_tab': [],
                    'channel_group': [],
                }
                grp['trigger'] = [trigger, trigger_text]
                grp['channel_dependencies'] = []

                if record_id_nr:
                    grp['sorted'] = False
                else:
                    grp['sorted'] = True

                kargs = {'first_cg_addr': cg_addr,
                         'data_block_addr': data_addr}
                if self.version >= '3.20':
                    kargs['block_len'] = v23c.DG_POST_320_BLOCK_SIZE
                else:
                    kargs['block_len'] = v23c.DG_PRE_320_BLOCK_SIZE
                kargs['record_id_nr'] = record_id_nr

                grp['data_group'] = DataGroup(**kargs)

                # read each channel group sequentially
                grp['channel_group'] = ChannelGroup(
                    address=cg_addr,
                    stream=stream,
                )

                # read name and comment for current channel group
                cg_texts = {}
                grp['texts']['channel_group'].append(cg_texts)

                address = grp['channel_group']['comment_addr']
                if address:
                    if memory != 'minimum':
                        block = TextBlock(
                            address=address,
                            stream=stream,
                        )
                        cg_texts['comment_addr'] = block
                    else:
                        cg_texts['comment_addr'] = address

                # go to first channel of the current channel group
                ch_addr = grp['channel_group']['first_ch_addr']
                ch_cntr = 0
                grp_chs = grp['channels']
                grp_conv = grp['channel_conversions']

                while ch_addr:
                    # read channel block and create channel object
                    new_ch = Channel(
                        address=ch_addr,
                        stream=stream,
                    )

                    # check if it has channel dependencies
                    if new_ch['ch_depend_addr']:
                        dep = ChannelDependency(
                            address=new_ch['ch_depend_addr'],
                            stream=stream,
                        )
                        grp['channel_dependencies'].append(dep)
                    else:
                        grp['channel_dependencies'].append(None)

                    # update channel map
                    ch_map[ch_addr] = (ch_cntr, dg_cntr)

                    # read conversion block
                    address = new_ch['conversion_addr']

                    if memory == 'minimum':
                        grp_conv.append(address)
                    else:
                        if address:
                            stream.seek(address + 2)
                            size = unpack('<H', stream.read(2))[0]
                            stream.seek(address)
                            raw_bytes = stream.read(size)

                            if raw_bytes in cc_map:
                                conv = cc_map[raw_bytes]
                            else:
                                conv = ChannelConversion(
                                    raw_bytes=raw_bytes,
                                    stream=stream,
                                    address=address,
                                )
                                cc_map[raw_bytes] = conv

                        else:
                            conv = None
                        grp_conv.append(conv)

                    address = new_ch['source_depend_addr']

                    if memory != 'minimum':
                        if address:
                            stream.seek(address)
                            raw_bytes = stream.read(v23c.CE_BLOCK_SIZE)

                            if raw_bytes in ce_map:
                                grp['channel_extensions'].append(
                                    ce_map[raw_bytes]
                                )
                            else:
                                block = ChannelExtension(
                                    raw_bytes=raw_bytes,
                                    stream=stream,
                                    address=address,
                                )
                                grp['channel_extensions'].append(block)
                                ce_map[raw_bytes] = block
                        else:
                            grp['channel_extensions'].append(None)
                    else:
                        grp['channel_extensions'].append(address)

                    # read text fields for channel
                    address = new_ch.get('long_name_addr', 0)
                    if address:
                        new_ch.name = name = get_text_v3(address, stream)
                    else:
                        new_ch.name = name = (
                            new_ch['short_name']
                            .decode('latin-1')
                            .strip(' \n\t\0')
                        )

                    address = new_ch.get('comment_addr', 0)
                    if address:
                        new_ch.comment = get_text_v3(address, stream)

                    address = new_ch.get('display_name_addr', 0)
                    if address:
                        display_name = get_text_v3(address, stream)
                        new_ch.display_name = display_name
                        if display_name in self.channels_db:
                            self.channels_db[display_name].append(
                                (dg_cntr, ch_cntr)
                            )
                        else:
                            self.channels_db[display_name] = []
                            self.channels_db[display_name].append(
                                (dg_cntr, ch_cntr)
                            )

                        # check if the source is included in the channel name
                        display_name = display_name.split('\\')
                        if len(display_name) > 1:
                            display_name = display_name[0]
                            if display_name in self.channels_db:
                                self.channels_db[display_name].append(
                                    (dg_cntr, ch_cntr)
                                )
                            else:
                                self.channels_db[display_name] = []
                                self.channels_db[display_name].append(
                                    (dg_cntr, ch_cntr)
                                )

                    if name in self.channels_db:
                        self.channels_db[name].append((dg_cntr, ch_cntr))
                    else:
                        self.channels_db[name] = []
                        self.channels_db[name].append((dg_cntr, ch_cntr))

                    # check if the source is included in the channel name
                    name = name.split('\\')
                    if len(name) > 1:
                        name = name[0]
                        if name in self.channels_db:
                            self.channels_db[name].append((dg_cntr, ch_cntr))
                        else:
                            self.channels_db[name] = []
                            self.channels_db[name].append((dg_cntr, ch_cntr))

                    if new_ch['channel_type'] == v23c.CHANNEL_TYPE_MASTER:
                        self.masters_db[dg_cntr] = ch_cntr
                    # go to next channel of the current channel group

                    ch_cntr += 1
                    if memory != 'minimum':
                        grp_chs.append(new_ch)
                    else:
                        grp_chs.append(ch_addr)
                    ch_addr = new_ch['next_ch_addr']

                cg_addr = grp['channel_group']['next_cg_addr']
                dg_cntr += 1

                current_cg_index += 1
                if self._callback:
                    self._callback(current_cg_index, cg_count)

                if self._terminate:
                    self.close()
                    return

            # store channel groups record sizes dict and data block size in
            # each new group data belong to the initial unsorted group, and
            # add the key 'sorted' with the value False to use a flag;
            # this is used later if memory=False

            cg_size = {}
            total_size = 0

            for grp in new_groups:
                record_id = grp['channel_group']['record_id']
                if PYVERSION == 2:
                    record_id = chr(record_id)
                cycles_nr = grp['channel_group']['cycles_nr']
                record_size = grp['channel_group']['samples_byte_nr']

                cg_size[record_id] = record_size

                record_size += record_id_nr
                total_size += record_size * cycles_nr

                grp['record_size'] = cg_size
                grp['size'] = total_size

            if memory == 'full':
                # read data block of the current data group
                dat_addr = data_group['data_block_addr']
                if dat_addr:
                    seek(dat_addr)
                    data = read(total_size)
                else:
                    data = b''
                if record_id_nr == 0:
                    grp = new_groups[0]
                    grp['data_location'] = v23c.LOCATION_MEMORY
                    grp['data_block'] = DataBlock(data=data)

                else:
                    # agregate data for each record ID in the cg_data dict
                    cg_data = defaultdict(list)
                    i = 0
                    size = len(data)
                    while i < size:
                        rec_id = data[i]
                        # skip record id
                        i += 1
                        rec_size = cg_size[rec_id]
                        rec_data = data[i: i + rec_size]
                        cg_data[rec_id].append(rec_data)
                        # possibly skip 2nd record id
                        if record_id_nr == 2:
                            i += rec_size + 1
                        else:
                            i += rec_size
                    for grp in new_groups:
                        grp['data_location'] = v23c.LOCATION_MEMORY
                        record_id = grp['channel_group']['record_id']
                        if PYVERSION == 2:
                            record_id = chr(record_id)
                        data = cg_data[record_id]
                        data = b''.join(data)
                        grp['channel_group']['record_id'] = 1
                        grp['data_block'] = DataBlock(data=data)
            else:
                for grp in new_groups:
                    grp['data_location'] = v23c.LOCATION_ORIGINAL_FILE
                    grp['data_group']['data_block_addr'] = data_group['data_block_addr']
                    grp['data_block_addr'] = [data_group['data_block_addr'], ]
                    grp['data_block_size'] = [total_size, ]

            self.groups.extend(new_groups)

            # go to next data group
            dg_addr = data_group['next_dg_addr']

        # finally update the channel depency references
        for grp in self.groups:
            for dep in grp['channel_dependencies']:
                if dep:
                    for i in range(dep['sd_nr']):
                        ref_channel_addr = dep['ch_{}'.format(i)]
                        channel = ch_map[ref_channel_addr]
                        dep.referenced_channels.append(channel)

        if self.memory == 'full':
            self.close()

    def configure(
            self,
            read_fragment_size=None,
            write_fragment_size=None,
            use_display_names=None,
            single_bit_uint_as_bool=None):
        """ configure read and write fragment size for chuncked
        data access

        Parameters
        ----------
        read_fragment_size : int
            size hint of splitted data blocks, default 8MB; if the initial size
            is smaller, then no data list is used. The actual split size
            depends on the data groups' records size
        write_fragment_size : int
            size hint of splitted data blocks, default 8MB; if the initial size
            is smaller, then no data list is used. The actual split size
            depends on the data groups' records size
        use_display_names : bool
            use display name if available for the Signal's name returned by the
            get method

        """

        if read_fragment_size is not None:
            self._read_fragment_size = int(read_fragment_size)

        if write_fragment_size:
            self._write_fragment_size = int(write_fragment_size)

        if use_display_names is not None:
            self._use_display_names = bool(use_display_names)

        if single_bit_uint_as_bool is not None:
            self._single_bit_uint_as_bool = bool(single_bit_uint_as_bool)

    def add_trigger(self,
                    group,
                    timestamp,
                    pre_time=0,
                    post_time=0,
                    comment=''):
        """ add trigger to data group

        Parameters
        ----------
        group : int
            group index
        timestamp : float
            trigger time
        pre_time : float
            trigger pre time; default 0
        post_time : float
            trigger post time; default 0
        comment : str
            trigger comment

        """
        comment_template = """<EVcomment>
    <TX>{}</TX>
</EVcomment>"""
        try:
            group = self.groups[group]
        except IndexError:
            return

        trigger, trigger_text = group['trigger']

        if comment:
            try:
                comment = ET.fromstring(comment)
                comment = comment.find('.//TX').text
            except:
                pass

        if trigger:
            count = trigger['trigger_events_nr']
            trigger['trigger_events_nr'] += 1
            trigger['block_len'] += 24
            trigger['trigger_{}_time'.format(count)] = timestamp
            trigger['trigger_{}_pretime'.format(count)] = pre_time
            trigger['trigger_{}_posttime'.format(count)] = post_time
            if comment:
                if trigger_text is None:
                    comment = '{}. {}'.format(
                        count + 1,
                        comment,
                    )
                    comment = comment_template.format(comment)
                    trigger_text = TextBlock(text=comment)
                    group['trigger'][1] = trigger_text
                else:
                    current_comment = trigger_text['text'].decode('latin-1').strip(' \r\t\n\0')
                    try:
                        current_comment = ET.fromstring(current_comment)
                        current_comment = current_comment.find('.//TX').text
                    except:
                        raise

                    comment = '{}\n{}. {}'.format(
                        current_comment,
                        count + 1,
                        comment,
                    )
                    comment = comment_template.format(comment)
                    trigger_text = TextBlock(text=comment)
                    group['trigger'][1] = trigger_text
        else:
            trigger = TriggerBlock(
                trigger_event_nr=1,
                trigger_0_time=timestamp,
                trigger_0_pretime=pre_time,
                trigger_0_posttime=post_time,
            )
            if comment:
                comment = '1. {}'.format(
                    comment,
                )
                comment = comment_template.format(comment)
                trigger_text = TextBlock(text=comment)
            else:
                trigger_text = None

            group['trigger'] = [trigger, trigger_text]

    def append(
            self,
            signals,
            acquisition_info='Python',
            common_timebase=False):
        """Appends a new data group.

        For channel dependencies type Signals, the *samples* attribute must be
        a numpy.recarray

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
            error = '"append" requires a non-empty list of Signal objects'
            raise MdfException(error)

        # check if the signals have a common timebase
        # if not interpolate the signals using the union of all timbases
        timestamps = signals[0].timestamps
        if not common_timebase:
            for signal in signals[1:]:
                if not array_equal(signal.timestamps, timestamps):
                    different = True
                    break
            else:
                different = False

            if different:
                times = [s.timestamps for s in signals]
                timestamps = reduce(union1d, times).flatten().astype(float64)
                signals = [s.interp(timestamps) for s in signals]
                times = None

        if self.version < '3.10':
            if timestamps.dtype.byteorder == '>':
                timestamps = timestamps.byteswap().newbyteorder()
            for signal in signals:
                if signal.samples.dtype.byteorder == '>':
                    signal.samples = signal.samples.byteswap().newbyteorder()

        if self.version >= '3.00':
            channel_size = v23c.CN_DISPLAYNAME_BLOCK_SIZE
        elif self.version >= '2.10':
            channel_size = v23c.CN_LONGNAME_BLOCK_SIZE
        else:
            channel_size = v23c.CN_SHORT_BLOCK_SIZE

        memory = self.memory
        file = self._tempfile
        write = file.write
        tell = file.tell

        kargs = {
            'module_nr': 0,
            'module_address': 0,
            'type': v23c.SOURCE_ECU,
            'description': b'Channel inserted by Python Script',
        }
        ce_block = ChannelExtension(**kargs)
        if memory == 'minimum':
            ce_address = tell()
            write(bytes(ce_block))

        if acquisition_info:
            acq_block = TextBlock(text=acquisition_info)
            if memory == 'minimum':
                acq_address = tell()
                write(bytes(acq_block))
        else:
            acq_block = None
            acq_address = 0

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

        dg_cntr = len(self.groups)

        gp = {}
        gp['channels'] = gp_channels = []
        gp['channel_conversions'] = gp_conv = []
        gp['channel_extensions'] = gp_source = []
        gp['channel_dependencies'] = gp_dep = []
        gp['texts'] = gp_texts = {
            'conversion_tab': [],
            'channel_group': [],
        }
        gp['signal_types'] = gp_sig_types = []
        gp['string_dtypes'] = []

        self.groups.append(gp)

        cycles_nr = len(timestamps)
        fields = []
        types = []
        parents = {}
        ch_cntr = 0
        offset = 0
        field_names = set()

        cg_texts = {}
        gp_texts['channel_group'].append(cg_texts)
        if memory == 'minimum':
            cg_texts['comment_addr'] = acq_address
        else:
            cg_texts['comment_addr'] = acq_block

        # time channel texts
        gp_texts['conversion_tab'].append(None)

        master_metadata = signals[0].master_metadata
        if master_metadata:
            time_name = master_metadata[0]
        else:
            time_name = 'Time'

        # conversion for time channel
        kargs = {
            'conversion_type': v23c.CONVERSION_TYPE_NONE,
            'unit': b's',
            'min_phy_value': timestamps[0] if cycles_nr else 0,
            'max_phy_value': timestamps[-1] if cycles_nr else 0,
        }
        block = ChannelConversion(**kargs)
        if memory != 'minimum':
            gp_conv.append(block)
        else:
            address = tell()
            gp_conv.append(address)
            write(bytes(block))

        # source for time
        if memory != 'minimum':
            gp_source.append(ce_block)
        else:
            gp_source.append(ce_address)

        # time channel
        t_type, t_size = fmt_to_datatype_v3(
            timestamps.dtype,
            timestamps.shape,
        )
        kargs = {
            'short_name': time_name.encode('latin-1'),
            'channel_type': v23c.CHANNEL_TYPE_MASTER,
            'data_type': t_type,
            'start_offset': 0,
            'min_raw_value': timestamps[0] if cycles_nr else 0,
            'max_raw_value': timestamps[-1] if cycles_nr else 0,
            'bit_count': t_size,
            'block_len': channel_size,
        }
        channel = Channel(**kargs)
        channel.name = name = time_name
        if memory != 'minimum':
            gp_channels.append(channel)
        else:
            address = tell()
            gp_channels.append(address)
            write(bytes(channel))

        if name not in self.channels_db:
            self.channels_db[name] = []
        self.channels_db[name].append((dg_cntr, ch_cntr))
        self.masters_db[dg_cntr] = 0
        # data group record parents
        parents[ch_cntr] = name, 0

        # time channel doesn't have channel dependencies
        gp_dep.append(None)

        fields.append(timestamps)
        types.append((name, timestamps.dtype))
        field_names.add(name)

        offset += t_size
        ch_cntr += 1

        gp_sig_types.append(0)

        for signal in signals:
            sig = signal
            names = sig.samples.dtype.names
            name = signal.name
            if names is None:
                sig_type = v23c.SIGNAL_TYPE_SCALAR
            else:
                if names in (canopen_time_fields, canopen_date_fields):
                    sig_type = v23c.SIGNAL_TYPE_CANOPEN
                elif names[0] != sig.name:
                    sig_type = v23c.SIGNAL_TYPE_STRUCTURE_COMPOSITION
                else:
                    sig_type = v23c.SIGNAL_TYPE_ARRAY

            gp_sig_types.append(sig_type)

            # conversions for channel

            conversion = conversion_transfer(signal.conversion)
            unit = signal.unit.strip(' \r\n\t\0')
            if unit:
                conversion['unit'] = unit.encode('latin-1')
            israw = signal.raw

            if not israw and not unit:
                if memory != 'minimum':
                    gp_conv.append(None)
                else:
                    gp_conv.append(0)
            else:

                if memory == 'minimum':
                    if conversion:
                        write_cc(
                            conversion,
                            defined_texts={},
                            stream=self._tempfile,
                        )
                        address = tell()
                        write(bytes(conversion))
                        gp_conv.append(address)
                    else:
                        gp_conv.append(0)
                else:
                    gp_conv.append(conversion)

            min_val, max_val = get_min_max(signal.samples)

            if sig_type == v23c.SIGNAL_TYPE_SCALAR:

                # source for channel
                if signal.source:
                    source = signal.source
                    if source.source_type != 2:
                        kargs = {
                            'type': v23c.SOURCE_ECU,
                            'description': source.name.encode('latin-1'),
                            'ECU_identification': source.path.encode('latin-1'),
                        }
                    else:
                        kargs = {
                            'type': v23c.SOURCE_VECTOR,
                            'message_name': source.name.encode('latin-1'),
                            'sender_name': source.path.encode('latin-1'),
                        }

                    new_source = ChannelExtension(**kargs)

                    if memory == 'minimum':
                        addr = tell()
                        write(bytes(new_source))
                        gp_source.append(addr)
                    else:
                        gp_source.append(new_source)
                else:
                    if memory != 'minimum':
                        gp_source.append(ce_block)
                    else:
                        gp_source.append(ce_address)

                # compute additional byte offset for large records size
                if offset > v23c.MAX_UINT16:
                    additional_byte_offset = ceil((offset - v23c.MAX_UINT16) / 8)
                    start_bit_offset = offset - additional_byte_offset * 8
                else:
                    start_bit_offset = offset
                    additional_byte_offset = 0

                s_type, s_size = fmt_to_datatype_v3(
                    signal.samples.dtype,
                    signal.samples.shape,
                )

                if (memory == 'minimum'
                        and len(name) >= 32
                        and self.version >= '2.10'):
                    block = TextBlock(text=name)
                    long_name_address = tell()
                    write(bytes(block))
                else:
                    long_name_address = 0
                comment = signal.comment
                if comment:
                    if len(comment) >= 128:
                        description = b'\0'
                        if memory == 'minimum':
                            block = TextBlock(text=comment)
                            comment_address = tell()
                            write(bytes(block))
                        else:
                            comment_address = 0
                    else:
                        description = (comment[:127] + '\0').encode('latin-1')
                        comment_address = 0
                else:
                    description = b'\0'
                    comment_address = 0
                display_name = signal.display_name
                if display_name:
                    if memory == 'minimum':
                        block = TextBlock(text=display_name)
                        display_name_address = tell()
                        write(bytes(block))
                    else:
                        display_name_address = 0
                else:
                    display_name_address = 0
                short_name = (name[:31] + '\0').encode('latin-1')

                if signal.samples.dtype.kind == 'u' and signal.bit_count <= 4:
                    s_size_ = signal.bit_count
                else:
                    s_size_ = s_size

                kargs = {
                    'short_name': short_name,
                    'channel_type': v23c.CHANNEL_TYPE_VALUE,
                    'data_type': s_type,
                    'min_raw_value': min_val if min_val <= max_val else 0,
                    'max_raw_value': max_val if min_val <= max_val else 0,
                    'start_offset': start_bit_offset,
                    'bit_count': s_size_,
                    'aditional_byte_offset': additional_byte_offset,
                    'long_name_addr': long_name_address,
                    'block_len': channel_size,
                    'comment_addr': comment_address,
                    'description': description,
                    'display_name_addr': display_name_address,
                }

                if s_size < 8:
                    s_size = 8

                channel = Channel(**kargs)
                if memory != 'minimum':
                    channel.name = name
                    channel.comment = comment
                    channel.display_name = display_name
                    gp_channels.append(channel)
                else:
                    address = tell()
                    gp_channels.append(address)
                    write(bytes(channel))
                offset += s_size

                if name not in self.channels_db:
                    self.channels_db[name] = []
                self.channels_db[name].append((dg_cntr, ch_cntr))

                # update the parents as well
                field_name = get_unique_name(field_names, name)
                parents[ch_cntr] = field_name, 0

                if signal.samples.dtype.kind == 'S':
                    gp['string_dtypes'].append(signal.samples.dtype)

                fields.append(signal.samples)
                if s_type != v23c.DATA_TYPE_BYTEARRAY:
                    types.append((field_name, signal.samples.dtype))
                else:
                    types.append((field_name, signal.samples.dtype, signal.samples.shape[1:]))
                field_names.add(field_name)

                # check if the source is included in the channel name
                name = name.split('\\')
                if len(name) > 1:
                    name = name[0]
                    if name in self.channels_db:
                        self.channels_db[name].append((dg_cntr, ch_cntr))
                    else:
                        self.channels_db[name] = []
                        self.channels_db[name].append((dg_cntr, ch_cntr))

                if display_name:
                    if display_name in self.channels_db:
                        self.channels_db[display_name].append((dg_cntr, ch_cntr))
                    else:
                        self.channels_db[display_name] = []
                        self.channels_db[display_name].append((dg_cntr, ch_cntr))

                    # check if the source is included in the channel name
                    display_name = display_name.split('\\')
                    if len(display_name) > 1:
                        display_name = display_name[0]
                        if display_name in self.channels_db:
                            self.channels_db[display_name].append((dg_cntr, ch_cntr))
                        else:
                            self.channels_db[display_name] = []
                            self.channels_db[display_name].append((dg_cntr, ch_cntr))

                ch_cntr += 1

                # simple channels don't have channel dependencies
                gp_dep.append(None)

            # second, add the composed signals
            elif sig_type in (
                    v23c.SIGNAL_TYPE_CANOPEN,
                    v23c.SIGNAL_TYPE_STRUCTURE_COMPOSITION):
                new_dg_cntr = len(self.groups)
                new_gp = {}
                new_gp['channels'] = new_gp_channels = []
                new_gp['channel_conversions'] = new_gp_conv = []
                new_gp['channel_extensions'] = new_gp_source = []
                new_gp['channel_dependencies'] = new_gp_dep = []
                new_gp['texts'] = new_gp_texts = {
                    'conversion_tab': [],
                    'channel_group': [],
                }
                self.groups.append(new_gp)

                new_fields = []
                new_types = []
                new_parents = {}
                new_ch_cntr = 0
                new_offset = 0
                new_field_names = set()

                # time channel texts
                new_gp_texts['conversion_tab'].append(None)

                # conversion for time channel
                kargs = {
                    'conversion_type': v23c.CONVERSION_TYPE_NONE,
                    'unit': b's',
                    'min_phy_value': timestamps[0] if cycles_nr else 0,
                    'max_phy_value': timestamps[-1] if cycles_nr else 0,
                }
                block = ChannelConversion(**kargs)
                if memory != 'minimum':
                    new_gp_conv.append(block)
                else:
                    address = tell()
                    new_gp_conv.append(address)
                    write(bytes(block))

                # source for time
                if memory != 'minimum':
                    new_gp_source.append(ce_block)
                else:
                    new_gp_source.append(ce_address)

                # time channel
                t_type, t_size = fmt_to_datatype_v3(
                    timestamps.dtype,
                    timestamps.shape,
                )
                kargs = {
                    'short_name': time_name.encode('latin-1'),
                    'channel_type': v23c.CHANNEL_TYPE_MASTER,
                    'data_type': t_type,
                    'start_offset': 0,
                    'min_raw_value': timestamps[0] if cycles_nr else 0,
                    'max_raw_value': timestamps[-1] if cycles_nr else 0,
                    'bit_count': t_size,
                    'block_len': channel_size,
                }
                channel = Channel(**kargs)
                channel.name = name = time_name
                if memory != 'minimum':
                    new_gp_channels.append(channel)
                else:
                    address = tell()
                    new_gp_channels.append(address)
                    write(bytes(channel))

                if name not in self.channels_db:
                    self.channels_db[name] = []
                self.channels_db[name].append((new_dg_cntr, new_ch_cntr))

                self.masters_db[new_dg_cntr] = 0
                # data group record parents
                new_parents[new_ch_cntr] = name, 0

                # time channel doesn't have channel dependencies
                new_gp_dep.append(None)

                new_fields.append(timestamps)
                new_types.append((name, timestamps.dtype))
                new_field_names.add(name)

                new_offset += t_size
                new_ch_cntr += 1

                # check if the source is included in the channel name
                name = name.split('\\')
                if len(name) > 1:
                    name = name[0]
                    if name in self.channels_db:
                        self.channels_db[name].append((dg_cntr, ch_cntr))
                    else:
                        self.channels_db[name] = []
                        self.channels_db[name].append((dg_cntr, ch_cntr))

                names = signal.samples.dtype.names
                if names == (
                        'ms',
                        'days'):
                    acq_block = TextBlock(text='From mdf v4 CANopen Time channel')
                    if memory == 'minimum':
                        acq_address = tell()
                        write(bytes(block))
                        new_gp_texts['channel_group'].append({'comment_addr': acq_address})
                    else:
                        new_gp_texts['channel_group'].append({'comment_addr': acq_block})
                elif names == (
                        'ms',
                        'min',
                        'hour',
                        'day',
                        'month',
                        'year',
                        'summer_time',
                        'day_of_week'):
                    acq_block = TextBlock(text='From mdf v4 CANopen Date channel')
                    if memory == 'minimum':
                        acq_address = tell()
                        write(bytes(block))
                        new_gp_texts['channel_group'].append({'comment_addr': acq_address})
                    else:
                        new_gp_texts['channel_group'].append({'comment_addr': acq_block})
                else:
                    text = 'From mdf v4 structure channel composition'
                    acq_block = TextBlock(text=text)
                    if memory == 'minimum':
                        acq_address = tell()
                        write(bytes(block))
                        new_gp_texts['channel_group'].append({'comment_addr': acq_address})
                    else:
                        new_gp_texts['channel_group'].append({'comment_addr': acq_block})

                for name in names:

                    samples = signal.samples[name]

                    new_gp_texts['conversion_tab'].append(None)

                    if memory == 'minimum' and len(name) >= 32 and self.version >= '2.10':
                        block = TextBlock(text=name)
                        long_name_address = tell()
                        write(bytes(block))
                    else:
                        long_name_address = 0
                    comment_address = 0
                    short_name = (name[:31] + '\0').encode('latin-1')

                    # conversions for channel
                    min_val, max_val = get_min_max(samples)

                    kargs = {
                        'conversion_type': v23c.CONVERSION_TYPE_NONE,
                        'unit': signal.unit.encode('latin-1'),
                        'min_phy_value': min_val if min_val <= max_val else 0,
                        'max_phy_value': max_val if min_val <= max_val else 0,
                    }
                    block = ChannelConversion(**kargs)
                    if memory != 'minimum':
                        new_gp_conv.append(block)
                    else:
                        address = tell()
                        new_gp_conv.append(address)
                        write(bytes(block))

                    # source for channel
                    if signal.source:
                        source = signal.source
                        if source.source_type != 2:
                            kargs = {
                                'type': v23c.SOURCE_ECU,
                                'description': source.name.encode('latin-1'),
                                'ECU_identification': source.path.encode('latin-1'),
                            }
                        else:
                            kargs = {
                                'type': v23c.SOURCE_VECTOR,
                                'message_name': source.name.encode('latin-1'),
                                'sender_name': source.path.encode('latin-1'),
                            }

                        new_source = ChannelExtension(**kargs)

                        if memory == 'minimum':
                            addr = tell()
                            write(bytes(new_source))
                            gp_source.append(addr)
                        else:
                            gp_source.append(new_source)
                    else:
                        if memory != 'minimum':
                            gp_source.append(ce_block)
                        else:
                            gp_source.append(ce_address)

                    # compute additional byte offset for large records size
                    if new_offset > v23c.MAX_UINT16:
                        additional_byte_offset = (new_offset - v23c.MAX_UINT16) >> 3
                        start_bit_offset = new_offset - additional_byte_offset << 3
                    else:
                        start_bit_offset = new_offset
                        additional_byte_offset = 0
                    s_type, s_size = fmt_to_datatype_v3(
                        samples.dtype,
                        samples.shape,
                    )

                    kargs = {
                        'short_name': short_name,
                        'channel_type': v23c.CHANNEL_TYPE_VALUE,
                        'data_type': s_type,
                        'min_raw_value': min_val if min_val <= max_val else 0,
                        'max_raw_value': max_val if min_val <= max_val else 0,
                        'start_offset': start_bit_offset,
                        'bit_count': s_size,
                        'aditional_byte_offset': additional_byte_offset,
                        'long_name_addr': long_name_address,
                        'block_len': channel_size,
                        'comment_addr': comment_address,
                    }

                    if s_size < 8:
                        s_size = 8

                    channel = Channel(**kargs)

                    if memory != 'minimum':
                        channel.name = name
                        new_gp_channels.append(channel)
                    else:
                        address = tell()
                        new_gp_channels.append(address)
                        write(bytes(channel))
                    new_offset += s_size

                    if name not in self.channels_db:
                        self.channels_db[name] = []
                    self.channels_db[name].append((new_dg_cntr, new_ch_cntr))

                    # update the parents as well
                    field_name = get_unique_name(new_field_names, name)
                    new_parents[new_ch_cntr] = field_name, 0

                    new_fields.append(samples)
                    new_types.append((field_name, samples.dtype))
                    new_field_names.add(field_name)

                    # check if the source is included in the channel name
                    name = name.split('\\')
                    if len(name) > 1:
                        name = name[0]
                        if name in self.channels_db:
                            self.channels_db[name].append((dg_cntr, ch_cntr))
                        else:
                            self.channels_db[name] = []
                            self.channels_db[name].append((dg_cntr, ch_cntr))

                    new_ch_cntr += 1

                    # simple channels don't have channel dependencies
                    new_gp_dep.append(None)

                # channel group
                kargs = {
                    'cycles_nr': cycles_nr,
                    'samples_byte_nr': new_offset >> 3,
                }
                new_gp['channel_group'] = ChannelGroup(**kargs)
                new_gp['channel_group']['ch_nr'] = new_ch_cntr
                new_gp['size'] = cycles_nr * (new_offset >> 3)

                # data group
                if self.version >= '3.20':
                    block_len = v23c.DG_POST_320_BLOCK_SIZE
                else:
                    block_len = v23c.DG_PRE_320_BLOCK_SIZE
                new_gp['data_group'] = DataGroup(block_len=block_len)

                # data block
                if PYVERSION == 2:
                    new_types = fix_dtype_fields(new_types)
                new_types = dtype(new_types)

                new_gp['types'] = new_types
                new_gp['parents'] = new_parents
                new_gp['sorted'] = True

                samples = fromarrays(new_fields, dtype=new_types)
                try:
                    block = samples.tostring()

                    if memory == 'full':
                        new_gp['data_location'] = v23c.LOCATION_MEMORY
                        kargs = {'data': block}
                        new_gp['data_block'] = DataBlock(**kargs)
                    else:
                        new_gp['data_location'] = v23c.LOCATION_TEMPORARY_FILE
                        if cycles_nr:
                            data_address = tell()
                            new_gp['data_group']['data_block_addr'] = data_address
                            self._tempfile.write(block)
                        else:
                            new_gp['data_group']['data_block_addr'] = 0
                except MemoryError:
                    if memory == 'full':
                        raise
                    else:
                        new_gp['data_location'] = v23c.LOCATION_TEMPORARY_FILE

                        data_address = tell()
                        new_gp['data_group']['data_block_addr'] = data_address
                        for sample in samples:
                            self._tempfile.write(sample.tostring())

                # data group trigger
                new_gp['trigger'] = [None, None]

            else:
                names = signal.samples.dtype.names
                name = signal.name

                component_names = []
                component_samples = []
                if names:
                    samples = signal.samples[names[0]]
                else:
                    samples = signal.samples

                shape = samples.shape[1:]
                dims = [list(range(size)) for size in shape]

                for indexes in product(*dims):
                    subarray = samples
                    for idx in indexes:
                        subarray = subarray[:, idx]
                    component_samples.append(subarray)

                    indexes = ''.join('[{}]'.format(idx) for idx in indexes)
                    component_name = '{}{}'.format(name, indexes)
                    component_names.append(component_name)

                # add channel dependency block for composed parent channel
                sd_nr = len(component_samples)
                kargs = {'sd_nr': sd_nr}
                for i, dim in enumerate(shape[::-1]):
                    kargs['dim_{}'.format(i)] = dim
                parent_dep = ChannelDependency(**kargs)
                gp_dep.append(parent_dep)

                gp_texts['conversion_tab'].append(None)

                # composed parent has no conversion
                if memory != 'minimum':
                    gp_conv.append(None)
                else:
                    gp_conv.append(0)

                # source for channel
                if signal.source:
                    source = signal.source
                    if source.source_type != 2:
                        kargs = {
                            'type': v23c.SOURCE_ECU,
                            'description': source.name.encode('latin-1'),
                            'ECU_identification': source.path.encode('latin-1'),
                        }
                    else:
                        kargs = {
                            'type': v23c.SOURCE_VECTOR,
                            'message_name': source.name.encode('latin-1'),
                            'sender_name': source.path.encode('latin-1'),
                        }

                    new_source = ChannelExtension(**kargs)

                    if memory == 'minimum':
                        addr = tell()
                        write(bytes(new_source))
                        gp_source.append(addr)
                    else:
                        gp_source.append(new_source)
                else:
                    if memory != 'minimum':
                        gp_source.append(ce_block)
                    else:
                        gp_source.append(ce_address)

                min_val, max_val = get_min_max(samples)

                s_type, s_size = fmt_to_datatype_v3(
                    samples.dtype,
                    (),
                    True,
                )
                # compute additional byte offset for large records size
                if offset > v23c.MAX_UINT16:
                    additional_byte_offset = (offset - v23c.MAX_UINT16) >> 3
                    start_bit_offset = offset - additional_byte_offset << 3
                else:
                    start_bit_offset = offset
                    additional_byte_offset = 0

                if memory == 'minimum' and len(name) >= 32 and self.version >= '2.10':
                    block = TextBlock(text=name)
                    long_name_address = tell()
                    write(bytes(block))
                else:
                    long_name_address = 0
                comment = signal.comment
                if comment:
                    if len(comment) >= 128:
                        description = b'\0'
                        if memory == 'minimum':
                            block = TextBlock(text=comment)
                            comment_address = tell()
                            write(bytes(block))
                        else:
                            comment_address = 0
                    else:
                        description = (comment[:127] + '\0').encode('latin-1')
                        comment_address = 0
                else:
                    description = b'\0'
                    comment_address = 0
                display_name = signal.display_name
                if display_name:
                    if memory == 'minimum':
                        block = TextBlock(text=display_name)
                        display_name_address = tell()
                        write(bytes(block))
                    else:
                        display_name_address = 0
                else:
                    display_name_address = 0

                short_name = (name[:31] + '\0').encode('latin-1')

                kargs = {
                    'short_name': short_name,
                    'channel_type': v23c.CHANNEL_TYPE_VALUE,
                    'data_type': s_type,
                    'min_raw_value': min_val if min_val <= max_val else 0,
                    'max_raw_value': max_val if min_val <= max_val else 0,
                    'start_offset': start_bit_offset,
                    'bit_count': s_size,
                    'aditional_byte_offset': additional_byte_offset,
                    'long_name_addr': long_name_address,
                    'block_len': channel_size,
                    'comment_addr': comment_address,
                    'description': description,
                    'display_name_addr': display_name_address,
                }

                if s_size <8:
                    s_size = 8

                channel = Channel(**kargs)
                if memory != 'minimum':
                    channel.name = name
                    channel.comment = comment
                    channel.display_name = display_name
                    gp_channels.append(channel)
                else:
                    address = tell()
                    gp_channels.append(address)
                    write(bytes(channel))

                if name not in self.channels_db:
                    self.channels_db[name] = []
                self.channels_db[name].append((dg_cntr, ch_cntr))

                # check if the source is included in the channel name
                name = name.split('\\')
                if len(name) > 1:
                    name = name[0]
                    if name in self.channels_db:
                        self.channels_db[name].append((dg_cntr, ch_cntr))
                    else:
                        self.channels_db[name] = []
                        self.channels_db[name].append((dg_cntr, ch_cntr))

                if display_name:
                    if display_name in self.channels_db:
                        self.channels_db[display_name].append((dg_cntr, ch_cntr))
                    else:
                        self.channels_db[display_name] = []
                        self.channels_db[display_name].append((dg_cntr, ch_cntr))

                    # check if the source is included in the channel name
                    display_name = display_name.split('\\')
                    if len(display_name) > 1:
                        display_name = display_name[0]
                        if display_name in self.channels_db:
                            self.channels_db[display_name].append((dg_cntr, ch_cntr))
                        else:
                            self.channels_db[display_name] = []
                            self.channels_db[display_name].append((dg_cntr, ch_cntr))

                ch_cntr += 1

                for i, (name, samples) in enumerate(
                        zip(component_names, component_samples)):
                    gp_texts['conversion_tab'].append(None)

                    if memory == 'minimum' and len(name) >= 32 and self.version >= '2.10':
                        block = TextBlock(text=name)
                        long_name_address = tell()
                        write(bytes(block))
                    else:
                        long_name_address = 0
                    if i < sd_nr:
                        dep_pair = ch_cntr, dg_cntr
                        parent_dep.referenced_channels.append(dep_pair)
                        description = b'\0'
                    else:
                        description = '{} - axis {}'.format(signal.name, name)
                        description = description.encode('latin-1')
                    comment_address = 0
                    short_name = (name[:31] + '\0').encode('latin-1')

                    min_val, max_val = get_min_max(samples)
                    s_type, s_size = fmt_to_datatype_v3(
                        samples.dtype,
                        (),
                    )
                    shape = samples.shape[1:]

                    # source for channel
                    if signal.source:
                        source = signal.source
                        if source.source_type != 2:
                            kargs = {
                                'type': v23c.SOURCE_ECU,
                                'description': source.name.encode('latin-1'),
                                'ECU_identification': source.path.encode('latin-1'),
                            }
                        else:
                            kargs = {
                                'type': v23c.SOURCE_VECTOR,
                                'message_name': source.name.encode('latin-1'),
                                'sender_name': source.path.encode('latin-1'),
                            }

                        new_source = ChannelExtension(**kargs)

                        if memory == 'minimum':
                            addr = tell()
                            write(bytes(new_source))
                            gp_source.append(addr)
                        else:
                            gp_source.append(new_source)
                    else:
                        if memory != 'minimum':
                            gp_source.append(ce_block)
                        else:
                            gp_source.append(ce_address)

                    if memory != 'minimum':
                        gp_conv.append(None)
                    else:
                        gp_conv.append(0)

                    # compute additional byte offset for large records size
                    if offset > v23c.MAX_UINT16:
                        additional_byte_offset = (offset - v23c.MAX_UINT16) >> 3
                        start_bit_offset = offset - additional_byte_offset << 3
                    else:
                        start_bit_offset = offset
                        additional_byte_offset = 0

                    kargs = {
                        'short_name': short_name,
                        'channel_type': v23c.CHANNEL_TYPE_VALUE,
                        'data_type': s_type,
                        'min_raw_value': min_val if min_val <= max_val else 0,
                        'max_raw_value': max_val if min_val <= max_val else 0,
                        'start_offset': start_bit_offset,
                        'bit_count': s_size,
                        'aditional_byte_offset': additional_byte_offset,
                        'long_name_addr': long_name_address,
                        'block_len': channel_size,
                        'comment_addr': comment_address,
                        'description': description,
                    }

                    if s_size < 8:
                        s_size = 8

                    channel = Channel(**kargs)
                    if memory != 'minimum':
                        channel.name = name
                        gp_channels.append(channel)
                    else:
                        address = tell()
                        gp_channels.append(address)
                        write(bytes(channel))

                    size = s_size
                    for dim in shape:
                        size *= dim
                    offset += size

                    if name not in self.channels_db:
                        self.channels_db[name] = []
                    self.channels_db[name].append((dg_cntr, ch_cntr))

                    # update the parents as well
                    field_name = get_unique_name(field_names, name)
                    parents[ch_cntr] = field_name, 0

                    fields.append(samples)
                    types.append((field_name, samples.dtype, shape))
                    field_names.add(field_name)

                    # check if the source is included in the channel name
                    name = name.split('\\')
                    if len(name) > 1:
                        name = name[0]
                        if name in self.channels_db:
                            self.channels_db[name].append((dg_cntr, ch_cntr))
                        else:
                            self.channels_db[name] = []
                            self.channels_db[name].append((dg_cntr, ch_cntr))

                    gp_dep.append(None)

                    ch_cntr += 1

                for name in names[1:]:
                    samples = signal.samples[name]

                    component_names = []
                    component_samples = []

                    shape = samples.shape[1:]
                    dims = [list(range(size)) for size in shape]

                    for indexes in product(*dims):
                        subarray = samples
                        for idx in indexes:
                            subarray = subarray[:, idx]
                        component_samples.append(subarray)

                        indexes = ''.join('[{}]'.format(idx) for idx in indexes)
                        component_name = '{}{}'.format(name, indexes)
                        component_names.append(component_name)

                    # add channel dependency block for composed parent channel
                    sd_nr = len(component_samples)
                    kargs = {'sd_nr': sd_nr}
                    for i, dim in enumerate(shape[::-1]):
                        kargs['dim_{}'.format(i)] = dim
                    parent_dep = ChannelDependency(**kargs)
                    gp_dep.append(parent_dep)

                    gp_texts['conversion_tab'].append(None)

                    # composed parent has no conversion
                    if memory != 'minimum':
                        gp_conv.append(None)
                    else:
                        gp_conv.append(0)

                    # source for channel
                    if signal.source:
                        source = signal.source
                        if source.source_type != 2:
                            kargs = {
                                'type': v23c.SOURCE_ECU,
                                'description': source.name.encode('latin-1'),
                                'ECU_identification': source.path.encode('latin-1'),
                            }
                        else:
                            kargs = {
                                'type': v23c.SOURCE_VECTOR,
                                'message_name': source.name.encode('latin-1'),
                                'sender_name': source.path.encode('latin-1'),
                            }

                        new_source = ChannelExtension(**kargs)

                        if memory == 'minimum':
                            addr = tell()
                            write(bytes(new_source))
                            gp_source.append(addr)
                        else:
                            gp_source.append(new_source)
                    else:
                        if memory != 'minimum':
                            gp_source.append(ce_block)
                        else:
                            gp_source.append(ce_address)

                    min_val, max_val = get_min_max(samples)

                    s_type, s_size = fmt_to_datatype_v3(
                        samples.dtype,
                        (),
                    )
                    # compute additional byte offset for large records size
                    if offset > v23c.MAX_UINT16:
                        additional_byte_offset = (offset - v23c.MAX_UINT16) >> 3
                        start_bit_offset = offset - additional_byte_offset << 3
                    else:
                        start_bit_offset = offset
                        additional_byte_offset = 0

                    if memory == 'minimum' and len(name) >= 32 and self.version >= '2.10':
                        block = TextBlock(text=name)
                        long_name_address = tell()
                        write(bytes(block))
                    else:
                        long_name_address = 0
                    comment = signal.comment
                    if comment:
                        if len(comment) >= 128:
                            description = b'\0'
                            if memory == 'minimum':
                                block = TextBlock(text=comment)
                                comment_address = tell()
                                write(bytes(block))
                            else:
                                comment_address = 0
                        else:
                            description = (comment[:127] + '\0').encode('latin-1')
                            comment_address = 0
                    else:
                        description = b'\0'
                        comment_address = 0
                    short_name = (name[:31] + '\0').encode('latin-1')

                    kargs = {
                        'short_name': short_name,
                        'channel_type': v23c.CHANNEL_TYPE_VALUE,
                        'data_type': s_type,
                        'min_raw_value': min_val if min_val <= max_val else 0,
                        'max_raw_value': max_val if min_val <= max_val else 0,
                        'start_offset': start_bit_offset,
                        'bit_count': s_size,
                        'aditional_byte_offset': additional_byte_offset,
                        'long_name_addr': long_name_address,
                        'block_len': channel_size,
                        'comment_addr': comment_address,
                        'description': description,
                    }

                    if s_size < 8:
                        s_size = 8

                    channel = Channel(**kargs)
                    if memory != 'minimum':
                        channel.name = name
                        channel.comment = signal.comment
                        gp_channels.append(channel)
                    else:
                        address = tell()
                        gp_channels.append(address)
                        write(bytes(channel))

                    if name not in self.channels_db:
                        self.channels_db[name] = []
                    self.channels_db[name].append((dg_cntr, ch_cntr))

                    # check if the source is included in the channel name
                    name = name.split('\\')
                    if len(name) > 1:
                        name = name[0]
                        if name in self.channels_db:
                            self.channels_db[name].append((dg_cntr, ch_cntr))
                        else:
                            self.channels_db[name] = []
                            self.channels_db[name].append((dg_cntr, ch_cntr))

                    ch_cntr += 1

                    for i, (name, samples) in enumerate(
                            zip(component_names, component_samples)):
                        gp_texts['conversion_tab'].append(None)

                        if memory == 'minimum' and len(name) >= 32 and self.version >= '2.10':
                            block = TextBlock(text=name)
                            long_name_address = tell()
                            write(bytes(block))
                        else:
                            long_name_address = 0
                        if i < sd_nr:
                            dep_pair = ch_cntr, dg_cntr
                            parent_dep.referenced_channels.append(dep_pair)
                            description = b'\0'
                        else:
                            description = '{} - axis {}'.format(signal.name, name)
                            description = description.encode('latin-1')
                        comment_address = 0
                        short_name = (name[:31] + '\0').encode('latin-1')

                        min_val, max_val = get_min_max(samples)
                        s_type, s_size = fmt_to_datatype_v3(
                            samples.dtype,
                            (),
                        )
                        shape = samples.shape[1:]

                        # source for channel
                        if signal.source:
                            source = signal.source
                            if source.source_type != 2:
                                kargs = {
                                    'type': v23c.SOURCE_ECU,
                                    'description': source.name.encode('latin-1'),
                                    'ECU_identification': source.path.encode('latin-1'),
                                }
                            else:
                                kargs = {
                                    'type': v23c.SOURCE_VECTOR,
                                    'message_name': source.name.encode('latin-1'),
                                    'sender_name': source.path.encode('latin-1'),
                                }

                            new_source = ChannelExtension(**kargs)

                            if memory == 'minimum':
                                addr = tell()
                                write(bytes(new_source))
                                gp_source.append(addr)
                            else:
                                gp_source.append(new_source)
                        else:
                            if memory != 'minimum':
                                gp_source.append(ce_block)
                            else:
                                gp_source.append(ce_address)

                        if memory != 'minimum':
                            gp_conv.append(None)
                        else:
                            gp_conv.append(0)

                        # compute additional byte offset for large records size
                        if offset > v23c.MAX_UINT16:
                            additional_byte_offset = (offset - v23c.MAX_UINT16) >> 3
                            start_bit_offset = offset - additional_byte_offset << 3
                        else:
                            start_bit_offset = offset
                            additional_byte_offset = 0

                        kargs = {
                            'short_name': short_name,
                            'channel_type': v23c.CHANNEL_TYPE_VALUE,
                            'data_type': s_type,
                            'min_raw_value': min_val if min_val <= max_val else 0,
                            'max_raw_value': max_val if min_val <= max_val else 0,
                            'start_offset': start_bit_offset,
                            'bit_count': s_size,
                            'aditional_byte_offset': additional_byte_offset,
                            'long_name_addr': long_name_address,
                            'block_len': channel_size,
                            'comment_addr': comment_address,
                            'description': description,
                        }

                        if s_size < 8:
                            s_size = 8

                        channel = Channel(**kargs)
                        if memory != 'minimum':
                            channel.name = name
                            gp_channels.append(channel)
                        else:
                            address = tell()
                            gp_channels.append(address)
                            write(bytes(channel))

                        size = s_size
                        for dim in shape:
                            size *= dim
                        offset += size

                        if name not in self.channels_db:
                            self.channels_db[name] = []
                        self.channels_db[name].append((dg_cntr, ch_cntr))

                        # update the parents as well
                        field_name = get_unique_name(field_names, name)
                        parents[ch_cntr] = field_name, 0

                        fields.append(samples)
                        types.append((field_name, samples.dtype, shape))
                        field_names.add(field_name)

                        # check if the source is included in the channel name
                        name = name.split('\\')
                        if len(name) > 1:
                            name = name[0]
                            if name in self.channels_db:
                                self.channels_db[name].append((dg_cntr, ch_cntr))
                            else:
                                self.channels_db[name] = []
                                self.channels_db[name].append((dg_cntr, ch_cntr))

                        gp_dep.append(None)

                        ch_cntr += 1

        # channel group
        kargs = {
            'cycles_nr': cycles_nr,
            'samples_byte_nr': offset >> 3,
        }
        if self.version >= '3.30':
            kargs['block_len'] = v23c.CG_POST_330_BLOCK_SIZE
        else:
            kargs['block_len'] = v23c.CG_PRE_330_BLOCK_SIZE
        gp['channel_group'] = ChannelGroup(**kargs)
        gp['channel_group']['ch_nr'] = ch_cntr
        gp['size'] = cycles_nr * (offset >> 3)

        # data group
        if self.version >= '3.20':
            block_len = v23c.DG_POST_320_BLOCK_SIZE
        else:
            block_len = v23c.DG_PRE_320_BLOCK_SIZE
        gp['data_group'] = DataGroup(block_len=block_len)

        # data block
        if PYVERSION == 2:
            types = fix_dtype_fields(types)
        types = dtype(types)

        gp['types'] = types
        gp['parents'] = parents
        gp['sorted'] = True

        samples = fromarrays(fields, dtype=types)

        block = samples.tostring()

        if memory == 'full':
            gp['data_location'] = v23c.LOCATION_MEMORY
            kargs = {'data': block}
            gp['data_block'] = DataBlock(**kargs)
        else:
            gp['data_location'] = v23c.LOCATION_TEMPORARY_FILE
            if cycles_nr:
                data_address = tell()
                gp['data_group']['data_block_addr'] = data_address
                gp['data_block_addr'] = [data_address, ]
                gp['data_block_size'] = [len(block), ]
                self._tempfile.write(block)
            else:
                gp['data_group']['data_block_addr'] = 0
                gp['data_block_addr'] = [0, ]
                gp['data_block_size'] = [0, ]

        # data group trigger
        gp['trigger'] = [None, None]

    def close(self):
        """ if the MDF was created with memory='minimum' and new
        channels have been appended, then this must be called just before the
        object is not used anymore to clean-up the temporary file

        """
        if self._tempfile is not None:
            self._tempfile.close()
        if self._file is not None:
            self._file.close()

    def extend(self, index, signals):
        """
        Extend a group with new samples. The first signal is the master channel's samples, and the
        next signals must respect the same order in which they were appended. The samples must have raw
        or physical values according to the *Signals* used for the initial append.

        Parameters
        ----------
        index : int
            group index
        signals : list
            list on numpy.ndarray objects

        Examples
        --------
        >>> # case 1 conversion type None
        >>> s1 = np.array([1, 2, 3, 4, 5])
        >>> s2 = np.array([-1, -2, -3, -4, -5])
        >>> s3 = np.array([0.1, 0.04, 0.09, 0.16, 0.25])
        >>> t = np.array([0.001, 0.002, 0.003, 0.004, 0.005])
        >>> names = ['Positive', 'Negative', 'Float']
        >>> units = ['+', '-', '.f']
        >>> s1 = Signal(samples=s1, timstamps=t, unit='+', name='Positive')
        >>> s2 = Signal(samples=s2, timstamps=t, unit='-', name='Negative')
        >>> s3 = Signal(samples=s3, timstamps=t, unit='flts', name='Floats')
        >>> mdf = MDF3('new.mdf')
        >>> mdf.append([s1, s2, s3], 'created by asammdf v1.1.0')
        >>> t = np.array([0.006, 0.007, 0.008, 0.009, 0.010])
        >>> mdf2.extend(0, [t, s1, s2, s3])

        """
        memory = self.memory
        new_group_offset = 0
        gp = self.groups[index]
        if not signals:
            message = '"append" requires a non-empty list of Signal objects'
            raise MdfException(message)

        if gp['data_location'] == v23c.LOCATION_ORIGINAL_FILE:
            stream = self._file
        else:
            stream = self._tempfile

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

        fields = []
        types = []

        cycles_nr = len(signals[0])
        string_counter = 0

        for signal in signals:
            sig = signal
            names = sig.dtype.names

            if len(sig.shape) <= 1:
                if names is None:
                    sig_type = v23c.SIGNAL_TYPE_SCALAR
                else:
                    if names in (canopen_time_fields, canopen_date_fields):
                        sig_type = v23c.SIGNAL_TYPE_CANOPEN
                    elif names[0] != sig.name:
                        sig_type = v23c.SIGNAL_TYPE_STRUCTURE_COMPOSITION
                    else:
                        sig_type = v23c.SIGNAL_TYPE_ARRAY
            else:
                sig_type = v23c.SIGNAL_TYPE_ARRAY

            if sig_type == v23c.SIGNAL_TYPE_SCALAR:
                if signal.dtype.kind == 'S':
                    str_dtype = gp['string_dtypes'][string_counter]
                    signal = signal.astype(str_dtype)
                    string_counter += 1
                fields.append(signal)

                if signal.shape[1:]:
                    types.append(('', signal.dtype, signal.shape[1:]))
                else:
                    types.append(('', signal.dtype))

            # second, add the composed signals
            elif sig_type in (
                    v23c.SIGNAL_TYPE_CANOPEN,
                    v23c.SIGNAL_TYPE_STRUCTURE_COMPOSITION):
                new_group_offset += 1
                new_gp = self.groups[index + new_group_offset]

                new_fields = []
                new_types = []

                names = signal.dtype.names
                for name in names:
                    new_fields.append(signal[name])
                    new_types.append(('', signal.dtype))

                # data block
                if PYVERSION == 2:
                    new_types = fix_dtype_fields(new_types)
                new_types = dtype(new_types)

                samples = fromarrays(new_fields, dtype=new_types)
                samples = samples.tostring()

                record_size = new_gp['channel_group']['samples_byte_nr']
                extended_size = cycles_nr * record_size
                new_gp['size'] += extended_size

                if memory == 'full':
                    if samples:
                        data = new_gp['data_block']['data'] + samples
                        new_gp['data_block'] = DataBlock(data=data)
                else:
                    if samples:
                        stream.seek(0, 2)
                        data_address = stream.tell()
                        new_gp['data_block_addr'].append(data_address)
                        new_gp['data_block_size'].append(extended_size)
                        stream.write(samples)

            else:

                names = signal.dtype.names

                component_samples = []
                if names:
                    samples = signal[names[0]]
                else:
                    samples = signal

                shape = samples.shape[1:]
                dims = [list(range(size)) for size in shape]

                for indexes in product(*dims):
                    subarray = samples
                    for idx in indexes:
                        subarray = subarray[:, idx]
                    component_samples.append(subarray)

                if names:
                    new_samples = [signal[fld] for fld in names[1:]]
                    component_samples.extend(new_samples)

                for samples in component_samples:
                    shape = samples.shape[1:]

                    fields.append(samples)
                    types.append(('', samples.dtype, shape))

        record_size = gp['channel_group']['samples_byte_nr']
        extended_size = cycles_nr * record_size
        gp['size'] += extended_size

        # data block
        if PYVERSION == 2:
            types = fix_dtype_fields(types)
        types = dtype(types)

        samples = fromarrays(fields, dtype=types)
        samples = samples.tostring()

        if memory == 'full':
            if samples:
                data = gp['data_block']['data'] + samples
                gp['data_block'] = DataBlock(data=data)
                gp['channel_group']['cycles_nr'] += cycles_nr
        else:
            if cycles_nr:
                stream.seek(0, 2)
                data_address = stream.tell()
                gp['data_block_addr'].append(data_address)
                gp['data_block_size'].append(extended_size)
                stream.write(samples)
                gp['channel_group']['cycles_nr'] += cycles_nr

    def get_channel_name(self, group, index):
        """Gets channel name.

        Parameters
        ----------
        group : int
            0-based group index
        index : int
            0-based channel index

        Returns
        -------
        name : str
            found channel name

        """
        gp_nr, ch_nr = self._validate_channel_selection(
            None,
            group,
            index,
        )

        grp = self.groups[gp_nr]
        if grp['data_location'] == v23c.LOCATION_ORIGINAL_FILE:
            stream = self._file
        else:
            stream = self._tempfile

        if self.memory == 'minimum':
            channel = Channel(
                address=grp['channels'][ch_nr],
                stream=stream,
            )
            address = channel.get('long_name_addr', 0)
            if address:
                name = get_text_v3(
                    address,
                    stream,
                )
            else:
                name = (
                    channel['short_name']
                    .decode('latin-1')
                    .strip(' \n\t\0')
                )

        else:
            name = grp['channels'][ch_nr].name

        return name

    def get_channel_metadata(
            self,
            name=None,
            group=None,
            index=None):
        gp_nr, ch_nr = self._validate_channel_selection(
            name,
            group,
            index,
        )

        grp = self.groups[gp_nr]

        if grp['data_location'] == v23c.LOCATION_ORIGINAL_FILE:
            stream = self._file
        else:
            stream = self._tempfile

        channel = grp['channels'][ch_nr]
        conversion = grp['channel_conversions'][ch_nr]
        source = grp['channel_extensions'][ch_nr]

        if self.memory == 'minimum':
            channel = Channel(
                address=channel,
                stream=stream,
            )

            channel.conversion = ChannelConversion(
                address=conversion,
                stream=stream,
            )

            channel.source = ChannelExtension(
                address=source,
                stream=stream,
            )

        else:
            channel = deepcopy(channel)
            channel.source = deepcopy(source)
            channel.conversion = deepcopy(conversion)

        return channel

    def get_channel_unit(self, name=None, group=None, index=None):
        """Gets channel unit.

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
        if grp['data_location'] == v23c.LOCATION_ORIGINAL_FILE:
            stream = self._file
        else:
            stream = self._tempfile

        if self.memory == 'minimum':
            addr = grp['channel_conversions'][ch_nr]
            if addr:
                conversion = ChannelConversion(
                    address=addr,
                    stream=stream,
                )
            else:
                conversion = None

        else:
            conversion = grp['channel_conversions'][ch_nr]

        if conversion:
            unit = (
                conversion['unit']
                .decode('latin-1')
                .strip(' \n\t\0')
            )
        else:
            unit = ''

        return unit

    def get_channel_comment(self, name=None, group=None, index=None):
        """Gets channel comment.
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
        if grp['data_location'] == v23c.LOCATION_ORIGINAL_FILE:
            stream = self._file
        else:
            stream = self._tempfile

        if self.memory == 'minimum':
            channel = Channel(
                address=grp['channels'][ch_nr],
                stream=stream,
            )
        else:
            channel = grp['channels'][ch_nr]

        if self.memory == 'minimum':
            comment = ''
            if channel['comment_addr']:
                comment = get_text_v3(channel['comment_addr'], stream)
        else:
            comment = channel.comment
        description = (
            channel['description']
            .decode('latin-1')
            .strip(' \t\n\0')
        )
        if comment:
            comment = '{}\n{}'.format(comment, description)
        else:
            comment = description

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
        interpolated accordingly.

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
            otherwise returns numpy.array.
            The *Signal* samples are:

                * numpy recarray for channels that have CDBLOCK or BYTEARRAY
                  type channels
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
        >>> mdf = MDF(version='3.30')
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

        original_data = data

        memory = self.memory
        grp = self.groups[gp_nr]

        if grp['data_location'] == v23c.LOCATION_ORIGINAL_FILE:
            stream = self._file
        else:
            stream = self._tempfile

        channel = grp['channels'][ch_nr]
        conversion = grp['channel_conversions'][ch_nr]

        if memory != 'minimum':
            channel = grp['channels'][ch_nr]
            conversion = grp['channel_conversions'][ch_nr]
            name = channel.name
            display_name = channel.display_name
        else:
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
                if channel.get('long_name_addr', 0):
                    name = get_text_v3(channel['long_name_addr'], stream)
                else:
                    name = (
                        channel['short_name']
                        .decode('latin-1')
                        .strip(' \n\t\0')
                    )
            channel.name = name
            if channel.get('display_name_addr', 0):
                display_name = get_text_v3(channel['display_name_addr'], stream)
            else:
                display_name = ''

        bit_count = channel['bit_count'] or 64

        dep = grp['channel_dependencies'][ch_nr]
        cycles_nr = grp['channel_group']['cycles_nr']

        # get data group record
        if data is None:
            data = self._load_group_data(grp)
        else:
            data = (data, )

        # check if this is a channel array
        if dep:
            if dep['dependency_type'] == v23c.DEPENDENCY_TYPE_VECTOR:
                shape = [dep['sd_nr'], ]
            elif dep['dependency_type'] >= v23c.DEPENDENCY_TYPE_NDIM:
                shape = []
                i = 0
                while True:
                    try:
                        dim = dep['dim_{}'.format(i)]
                        shape.append(dim)
                        i += 1
                    except KeyError:
                        break
                shape = shape[::-1]

            record_shape = tuple(shape)

            arrays = [
                self.get(group=dg_nr, index=ch_nr, samples_only=True, raw=raw, data=original_data)
                for ch_nr, dg_nr in dep.referenced_channels
            ]
            if cycles_nr:
                shape.insert(0, cycles_nr)

            vals = column_stack(arrays).flatten().reshape(tuple(shape))

            arrays = [vals, ]
            types = [(channel.name, vals.dtype, record_shape), ]

            if PYVERSION == 2:
                types = fix_dtype_fields(types)

            types = dtype(types)
            vals = fromarrays(arrays, dtype=types)

            if not samples_only or raster:
                timestamps = self.get_master(gp_nr, original_data)
                if raster and len(timestamps):
                    t = arange(
                        timestamps[0],
                        timestamps[-1],
                        raster,
                    )

                    vals = Signal(
                        vals,
                        timestamps,
                        name='_',
                    ).interp(t).samples

                    timestamps = t

        else:
            # get channel values
            channel_values = []
            timestamps = []
            count = 0
            for fragment in data:
                data_bytes, _ = fragment
                try:
                    parents, dtypes = grp['parents'], grp['types']
                except KeyError:
                    grp['parents'], grp['types'] = self._prepare_record(grp)
                    parents, dtypes = grp['parents'], grp['types']

                try:
                    parent, bit_offset = parents[ch_nr]
                except KeyError:
                    parent, bit_offset = None, None

                bits = channel['bit_count']

                if parent is not None:
                    if 'record' not in grp:
                        if dtypes.itemsize:
                            record = fromstring(data_bytes, dtype=dtypes)
                        else:
                            record = None

                        if memory == 'full':
                            grp['record'] = record
                    else:
                        record = grp['record']

                    record.setflags(write=False)

                    vals = record[parent]
                    data_type = channel['data_type']
                    size = vals.dtype.itemsize
                    if data_type == v23c.DATA_TYPE_BYTEARRAY:
                        size *= vals.shape[1]

                    vals_dtype = vals.dtype.kind
                    if vals_dtype not in 'ui' and (bit_offset or not bits == size * 8):
                        vals = self._get_not_byte_aligned_data(data_bytes, grp, ch_nr)
                    else:
                        if bit_offset:
                            dtype_ = vals.dtype
                            if dtype_.kind == 'i':
                                vals = vals.astype(dtype('<u{}'.format(size)))
                                vals >>= bit_offset
                            else:
                                vals = vals >> bit_offset

                        if not bits == size * 8:
                            if data_type in v23c.SIGNED_INT:
                                vals = as_non_byte_sized_signed_int(vals, bits)
                            else:
                                mask = (1 << bits) - 1
                                if vals.flags.writeable:
                                    vals &= mask
                                else:
                                    vals = vals & mask

                else:
                    vals = self._get_not_byte_aligned_data(data_bytes, grp, ch_nr)

                if not samples_only or raster:
                    timestamps.append(self.get_master(gp_nr, fragment))

                if bits == 1 and self._single_bit_uint_as_bool:
                    vals = array(vals, dtype=bool)
                else:
                    data_type = channel['data_type']
                    channel_dtype = array([], dtype=get_fmt_v3(data_type, bits))
                    if vals.dtype != channel_dtype.dtype:
                        vals = vals.astype(channel_dtype.dtype)

                channel_values.append(vals.copy())
                count += 1

            if count > 1:
                vals = concatenate(channel_values)
            elif count == 1:
                vals = channel_values[0]
            else:
                vals = []

            if not samples_only or raster:
                if count > 1:
                    timestamps = concatenate(timestamps)
                else:
                    timestamps = timestamps[0]

                if raster and len(timestamps):
                    t = arange(
                        timestamps[0],
                        timestamps[-1],
                        raster,
                    )

                    vals = Signal(
                        vals,
                        timestamps,
                        name='_',
                    ).interp(t).samples

                    timestamps = t

            if conversion is None:
                conversion_type = v23c.CONVERSION_TYPE_NONE
            else:
                conversion_type = conversion['conversion_type']

            if conversion_type == v23c.CONVERSION_TYPE_NONE:
                pass

            elif conversion_type in (
                    v23c.CONVERSION_TYPE_LINEAR,
                    v23c.CONVERSION_TYPE_TABI,
                    v23c.CONVERSION_TYPE_TAB,
                    v23c.CONVERSION_TYPE_EXPO,
                    v23c.CONVERSION_TYPE_LOGH,
                    v23c.CONVERSION_TYPE_RAT,
                    v23c.CONVERSION_TYPE_POLY,
                    v23c.CONVERSION_TYPE_FORMULA):
                if not raw:
                    vals = conversion.convert(vals)

            elif conversion_type in (
                    v23c.CONVERSION_TYPE_TABX,
                    v23c.CONVERSION_TYPE_RTABX):
                raw = True

        if samples_only:
            res = vals
        else:
            if conversion:
                unit = conversion['unit'].decode('latin-1').strip(' \n\r\t\0')
            else:
                unit = ''

            if memory == 'minimum':
                comment = ''
                if channel['comment_addr']:
                    comment = get_text_v3(channel['comment_addr'], stream)
            else:
                comment = channel.comment
            description = (
                channel['description']
                .decode('latin-1')
                .strip(' \t\n\0')
            )
            if comment:
                comment = '{}\n{}'.format(comment, description)
            else:
                comment = description

            if memory == 'minimum':
                addr = grp['channel_extensions'][ch_nr]
                if addr:
                    source = ChannelExtension(
                        address=addr,
                        stream=stream,
                    )
                else:
                    source = None
            else:
                source = grp['channel_extensions'][ch_nr]

            if source:
                if source['type'] == v23c.SOURCE_ECU:
                    source = SignalSource(
                        source.name,
                        source.path,
                        source.comment,
                        0, # source type other
                        0, # bus type none
                    )
                else:
                    source = SignalSource(
                        source.name,
                        source.path,
                        source.comment,
                        2,  # source type bus
                        2,  # bus type CAN
                    )

            master_metadata = self._master_channel_metadata.get(gp_nr, None)

            if display_name and not comment.startswith('<CNcomment'):
                CNcomment = ET.Element('CNcomment')

                tx = ET.Element('TX')
                tx.text = comment
                CNcomment.append(tx)

                display = ET.Element('display')
                display.text = display_name
                names = ET.Element('names')
                names.append(display)
                CNcomment.append(names)

                comment = ET.tostring(CNcomment).decode('utf-8')

            res = Signal(
                samples=vals,
                timestamps=timestamps,
                unit=unit,
                name=channel.name,
                comment=comment,
                conversion=conversion,
                raw=raw,
                master_metadata=master_metadata,
                display_name=display_name,
                source=source,
                bit_count=bit_count,
            )

        return res

    def get_master(self, index, data=None, raster=None):
        """ returns master channel samples for given group

        Parameters
        ----------
        index : int
            group index
        data : (bytes, int)
            (data block raw bytes, fragment offset); default None
        raster : float
            raster to be used for interpolation; default None

        Returns
        -------
        t : numpy.array
            master channel samples

        """
        original_data = data

        fragment = data
        if fragment:
            data_bytes, offset = fragment
            try:
                timestamps = self._master_channel_cache[(index, offset)]
                if raster and timestamps:
                    timestamps = arange(
                        timestamps[0],
                        timestamps[-1],
                        raster,
                    )
                return timestamps
            except KeyError:
                pass
        else:
            try:
                timestamps = self._master_channel_cache[index]
                if raster and timestamps:
                    timestamps = arange(
                        timestamps[0],
                        timestamps[-1],
                        raster,
                    )
                return timestamps
            except KeyError:
                pass

        group = self.groups[index]

        if group['data_location'] == v23c.LOCATION_ORIGINAL_FILE:
            stream = self._file
        else:
            stream = self._tempfile
        memory = self.memory

        time_ch_nr = self.masters_db.get(index, None)
        cycles_nr = group['channel_group']['cycles_nr']

        if time_ch_nr is None:
            t = arange(cycles_nr, dtype=float64)
            metadata = (
                'Time',
                1,
            )
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
                time_name = self.get_channel_name(
                    index,
                    time_ch_nr,
                )
            else:
                time_ch = group['channels'][time_ch_nr]
                time_name = time_ch.name

            metadata = (
                time_name,
                1,
            )

            if time_ch['bit_count'] == 0:
                if time_ch['sampling_rate']:
                    sampling_rate = time_ch['sampling_rate']
                else:
                    sampling_rate = 1
                t = arange(cycles_nr, dtype=float64) * sampling_rate
            else:
                # get data group parents and dtypes
                try:
                    parents, dtypes = group['parents'], group['types']
                except KeyError:
                    parents, dtypes = self._prepare_record(group)
                    group['parents'], group['types'] = parents, dtypes

                # get data group record
                if data is None:
                    data = self._load_group_data(group)
                else:
                    data = (data, )

                time_values = []
                count = 0
                for fragment in data:
                    data_bytes, offset = fragment
                    parent, _ = parents.get(time_ch_nr, (None, None))
                    if parent is not None:
                        not_found = object()
                        record = group.get('record', not_found)
                        if record is not_found:
                            if dtypes.itemsize:
                                record = fromstring(data_bytes, dtype=dtypes)
                            else:
                                record = None

                            if memory == 'full':
                                group['record'] = record
                        record.setflags(write=False)
                        t = record[parent]
                    else:
                        t = self._get_not_byte_aligned_data(
                            data_bytes,
                            group,
                            time_ch_nr,
                        )
                    time_values.append(t.copy())
                    count += 1

                if count > 1:
                    t = concatenate(time_values)
                elif count == 1:
                    t = time_values[0]
                else:
                    t = array([], dtype=float64)

                # get timestamps
                if time_conv is None:
                    time_conv_type = v23c.CONVERSION_TYPE_NONE
                else:
                    time_conv_type = time_conv['conversion_type']
                if time_conv_type == v23c.CONVERSION_TYPE_LINEAR:
                    time_a = time_conv['a']
                    time_b = time_conv['b']
                    t = t * time_a
                    if time_b:
                        t += time_b

        if not t.dtype == float64:
            t = t.astype(float64)

        self._master_channel_metadata[index] = metadata

        if original_data is None:
            self._master_channel_cache[index] = t
        else:
            data_bytes, offset = original_data
            self._master_channel_cache[(index, offset)] = t

        if raster and t.size:
            timestamps = arange(
                t[0],
                t[-1],
                raster,
            )
        else:
            timestamps = t
        return timestamps

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
                    comment = trigger_text['text'].decode('latin-1')
                    comment = comment.strip(' \n\t\0')
                else:
                    comment = ''

                for j in range(trigger['trigger_events_nr']):
                    trigger_info = {
                        'comment': comment,
                        'index': j,
                        'group': i,
                        'time': trigger['trigger_{}_time'.format(j)],
                        'pre_time': trigger['trigger_{}_pretime'.format(j)],
                        'post_time': trigger['trigger_{}_posttime'.format(j)],
                    }
                    yield trigger_info

    def info(self):
        """get MDF information as a dict

        Examples
        --------
        >>> mdf = MDF3('test.mdf')
        >>> mdf.info()

        """
        info = {}
        for key in ('author',
                    'organization',
                    'project',
                    'subject'):
            value = self.header[key].decode('latin-1').strip(' \n\t\0')
            info[key] = value
        info['version'] = self.version
        info['groups'] = len(self.groups)
        for i, gp in enumerate(self.groups):
            if gp['data_location'] == v23c.LOCATION_ORIGINAL_FILE:
                stream = self._file
            elif gp['data_location'] == v23c.LOCATION_TEMPORARY_FILE:
                stream = self._tempfile
            inf = {}
            info['group {}'.format(i)] = inf
            inf['cycles'] = gp['channel_group']['cycles_nr']
            inf['channels count'] = len(gp['channels'])
            for j, channel in enumerate(gp['channels']):
                if self.memory != 'minimum':
                    name = channel.name
                else:
                    channel = Channel(
                        address=channel,
                        stream=stream,
                    )
                    if channel.get('long_name_addr', 0):
                        name = get_text_v3(channel['long_name_addr'], stream)
                    else:
                        name = (
                            channel['short_name']
                            .decode('utf-8')
                            .strip(' \r\t\n\0')
                        )

                if channel['channel_type'] == v23c.CHANNEL_TYPE_MASTER:
                    ch_type = 'master'
                else:
                    ch_type = 'value'
                inf['channel {}'.format(j)] = 'name="{}" type={}'.format(
                    name,
                    ch_type,
                )

        return info

    def save(self, dst='', overwrite=False, compression=0):
        """Save MDF to *dst*. If *dst* is not provided the the destination file
        name is the MDF name. If overwrite is *True* then the destination file
        is overwritten, otherwise the file name is appended with '_<cntr>',
        were '<cntr>' is the first counter that produces a new file name (that
        does not already exist in the filesystem).

        Parameters
        ----------
        dst : str
            destination file name, Default ''
        overwrite : bool
            overwrite flag, default *False*
        compression : int
            does nothing for mdf version3; introduced here to share the same
            API as mdf version 4 files

        Returns
        -------
        output_file : str
            output file name

        """

        if self.name is None and dst == '':
            message = ('Must specify a destination file name '
                       'for MDF created from scratch')
            raise MdfException(message)

        _read_fragment_size = self._read_fragment_size
        self.configure(read_fragment_size=4 * 2**20)

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

        self.configure(read_fragment_size=_read_fragment_size)

        if self._callback:
            self._callback(100, 100)

        return output_file

    def _save_with_metadata(self, dst, overwrite, compression):
        """Save MDF to *dst*. If *dst* is not provided the the destination file
        name is the MDF name. If overwrite is *True* then the destination file
        is overwritten, otherwise the file name is appended with '_<cntr>',
        were '<cntr>' is the first counter that produces a new file name (that
        does not already exist in the filesystem).

        Parameters
        ----------
        dst : str
            destination file name, Default ''
        overwrite : bool
            overwrite flag, default *False*
        compression : int
            does nothing for mdf version3; introduced here to share the same
            API as mdf version 4 files

        """
        # pylint: disable=unused-argument

        if self.file_history is None:
            self.file_history = TextBlock(text='''<FHcomment>
<TX>created</TX>
<tool_id>asammdf</tool_id>
<tool_vendor> </tool_vendor>
<tool_version>{}</tool_version>
</FHcomment>'''.format(__version__))
        else:
            text = '{}\n{}: updated by asammdf {}'
            old_history = self.file_history['text'].decode('latin-1')
            timestamp = time.asctime().encode('latin-1')

            text = text.format(
                old_history,
                timestamp,
                __version__,
            )
            self.file_history = TextBlock(text=text)

        if self.name is None and dst == '':
            message = (
                'Must specify a destination file name '
                'for MDF created from scratch'
            )
            raise MdfException(message)

        dst = dst if dst else self.name
        if not dst.endswith(('mdf', 'MDF')):
            dst = dst + '.mdf'
        if overwrite is False:
            if os.path.isfile(dst):
                cntr = 0
                while True:
                    name = os.path.splitext(dst)[0] + '_{}.mdf'.format(cntr)
                    if not os.path.isfile(name):
                        break
                    else:
                        cntr += 1
                message = (
                    'Destination file "{}" already exists '
                    'and "overwrite" is False. Saving MDF file as "{}"'
                )
                message = message.format(dst, name)
                logger.warning(message)
                dst = name

        # all MDF blocks are appended to the blocks list in the order in which
        # they will be written to disk. While creating this list, all the
        # relevant block links are updated so that once all blocks have been
        # added to the list they can be written using the bytes protocol.
        # DataGroup blocks are written first after the identification and
        # header blocks. When memory='low' we need to restore the
        # original data block addresses within the data group block. This is
        # needed to allow further work with the object after the save method
        # call (eq. new calls to get method). Since the data group blocks are
        # written first, it is safe to restor the original links when the data
        # blocks are written. For memory=False the blocks list will
        # contain a tuple instead of a DataBlock instance; the tuple will have
        # the reference to the data group object and the original link to the
        # data block in the soource MDF file.

        if self.memory == 'low' and dst == self.name:
            destination = dst + '.temp'
        else:
            destination = dst

        with open(destination, 'wb+') as dst_:
            defined_texts = {}

            groups_nr = len(self.groups)

            write = dst_.write
            seek = dst_.seek
            # list of all blocks
            blocks = []

            address = 0

            write(bytes(self.identification))
            address += v23c.ID_BLOCK_SIZE

            write(bytes(self.header))
            address += self.header['block_len']

            if self.header.program:
                write(bytes(self.header.program))
                self.header['program_addr'] = address
                address += self.header.program['block_len']
            else:
                self.header['program_addr'] = 0

            self.file_history.address = address
            write(bytes(self.file_history))
            address += self.file_history['block_len']

            ce_map = {}

            # DataGroup
            # put them first in the block list so they will be written first to
            # disk this way, in case of memory=False, we can safely
            # restore he original data block address
            gp_rec_ids = []

            original_data_block_addrs = [
                group['data_group']['data_block_addr']
                for group in self.groups
            ]

            for idx, gp in enumerate(self.groups):
                dg = gp['data_group']
                gp_rec_ids.append(dg['record_id_nr'])
                dg['record_id_nr'] = 0

                # DataBlock
                for (data_bytes, _) in self._load_group_data(gp):
                    if self.memory == 'full':
                        data = memoryview(data_bytes)
                        read_size = 4 * 2**20
                        count = int(ceil(len(data_bytes) / read_size))
                        for j in range(count):
                            write(data[j*read_size: (j+1)*read_size])
                    else:
                        write(data_bytes)

                if gp['size']:
                    gp['data_group']['data_block_addr'] = address
                else:
                    gp['data_group']['data_block_addr'] = 0
                address += gp['size'] - gp_rec_ids[idx] * gp['channel_group']['cycles_nr']

                if self._callback:
                    self._callback(int(33 * (idx+1) / groups_nr), 100)
                if self._terminate:
                    dst_.close()
                    self.close()
                    return

            for gp in self.groups:
                dg = gp['data_group']
                blocks.append(dg)
                dg.address = address
                address += dg['block_len']

            if self.groups:
                for i, dg in enumerate(self.groups[:-1]):
                    addr = self.groups[i + 1]['data_group'].address
                    dg['data_group']['next_dg_addr'] = addr
                self.groups[-1]['data_group']['next_dg_addr'] = 0

            for idx, gp in enumerate(self.groups):
                gp_texts = gp['texts']

                # Texts
                for item_list in gp_texts.values():
                    for my_dict in item_list:
                        if my_dict is None:
                            continue
                        for tx_block in my_dict.values():
                            # text blocks can be shared
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
                    if conv is None:
                        continue
                    else:
                        blocks.append(conv)
                        conv.address = address
                        address += conv['block_len']
                        address = write_cc(conv, defined_texts, blocks, address)

                # Channel Extension
                cs = gp['channel_extensions']
                for source in cs:
                    if source:
                        source_id = id(source)
                        if source_id not in ce_map:
                            source.address = address
                            ce_map[source_id] = source
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
                for i, channel in enumerate(gp['channels']):
                    channel.address = address
                    blocks.append(channel)
                    address += channel['block_len']

                    comment = channel.comment
                    if comment:
                        if len(comment) >= 128:
                            channel['description'] = b'\0'
                            tx_block = TextBlock(text=comment)
                            text = tx_block['text']
                            if text in defined_texts:
                                channel['comment_addr'] = defined_texts[text].address
                            else:
                                channel['comment_addr'] = address
                                defined_texts[text] = tx_block
                                tx_block.address = address
                                blocks.append(tx_block)
                                address += tx_block['block_len']
                        else:
                            channel['description'] = (comment[:127] + '\0').encode('latin-1')
                            channel['comment_addr'] = 0
                    if self.version >= '2.10':
                        if len(channel.name) >= 32:
                            tx_block = TextBlock(text=channel.name)
                            text = tx_block['text']
                            if text in defined_texts:
                                channel['long_name_addr'] = defined_texts[text].address
                            else:
                                channel['long_name_addr'] = address
                                defined_texts[text] = tx_block
                                tx_block.address = address
                                blocks.append(tx_block)
                                address += tx_block['block_len']
                        else:
                            channel['long_name_addr'] = 0
                    if 'display_name_addr' in channel:
                        if channel.display_name:
                            tx_block = TextBlock(text=channel.display_name)
                            text = tx_block['text']
                            if text in defined_texts:
                                channel['display_name_addr'] = defined_texts[text].address
                            else:
                                channel['display_name_addr'] = address
                                defined_texts[text] = tx_block
                                tx_block.address = address
                                blocks.append(tx_block)
                                address += tx_block['block_len']
                        else:
                            channel['display_name_addr'] = 0

                    channel['conversion_addr'] = cc[i].address if cc[i] else 0
                    if cs[i]:
                        channel['source_depend_addr'] = cs[i].address
                    else:
                        channel['source_depend_addr'] = 0
                    if cd[i]:
                        channel['ch_depend_addr'] = cd[i].address
                    else:
                        channel['ch_depend_addr'] = 0

                count = len(gp['channels'])
                if count:
                    for i in range(count-1):
                        gp['channels'][i]['next_ch_addr'] = gp['channels'][i+1].address
                    gp['channels'][-1]['next_ch_addr'] = 0

                # ChannelGroup
                cg = gp['channel_group']
                cg.address = address
                blocks.append(cg)
                address += cg['block_len']

                cg['first_ch_addr'] = gp['channels'][0].address
                cg['next_cg_addr'] = 0
                cg_texts = gp['texts']['channel_group'][0]
                if 'comment_addr' in cg_texts:
                    addr = cg_texts['comment_addr'].address
                    cg['comment_addr'] = addr

                # TriggerBLock
                trigger, trigger_text = gp['trigger']
                if trigger:
                    if trigger_text:
                        trigger_text.address = address
                        blocks.append(trigger_text)
                        address += trigger_text['block_len']
                        trigger['text_addr'] = trigger_text.address
                    else:
                        trigger['text_addr'] = 0

                    trigger.address = address
                    blocks.append(trigger)
                    address += trigger['block_len']

                if self._callback:
                    self._callback(int(33 * (idx+1) / groups_nr) + 33, 100)
                if self._terminate:
                    dst_.close()
                    self.close()
                    return

            # update referenced channels addresses in the channel dependecies
            for gp in self.groups:
                for dep in gp['channel_dependencies']:
                    if not dep:
                        continue

                    for i, pair_ in enumerate(dep.referenced_channels):
                        ch_nr, dg_nr = pair_
                        grp = self.groups[dg_nr]
                        ch = grp['channels'][ch_nr]
                        dep['ch_{}'.format(i)] = ch.address
                        dep['cg_{}'.format(i)] = grp['channel_group'].address
                        dep['dg_{}'.format(i)] = grp['data_group'].address

            # DataGroup
            for gp in self.groups:
                gp['data_group']['first_cg_addr'] = gp['channel_group'].address
                if gp['trigger'][0]:
                    gp['data_group']['trigger_addr'] = gp['trigger'][0].address
                else:
                    gp['data_group']['trigger_addr'] = 0

            if self.groups:
                address = self.groups[0]['data_group'].address
                self.header['first_dg_addr'] = address
                self.header['dg_nr'] = len(self.groups)
                self.header['comment_addr'] = self.file_history.address

            if self._terminate:
                dst_.close()
                self.close()
                return

            if self._callback:
                blocks_nr = len(blocks)
                threshold = blocks_nr / 33
                count = 1
                for i, block in enumerate(blocks):
                    write(bytes(block))
                    if i >= threshold:
                        self._callback(66 + count, 100)
                        count += 1
                        threshold += blocks_nr / 33
            else:
                for block in blocks:
                    write(bytes(block))

            for gp, rec_id, original_address in zip(
                    self.groups,
                    gp_rec_ids,
                    original_data_block_addrs):
                gp['data_group']['record_id_nr'] = rec_id
                gp['data_group']['data_block_addr'] = original_address

            seek(0)
            write(bytes(self.identification))
            write(bytes(self.header))

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

            self._master_channel_cache = {}

            self._tempfile = TemporaryFile()
            self._file = open(self.name, 'rb')
            self._read()
        return dst

    def _save_without_metadata(self, dst, overwrite, compression):
        """Save MDF to *dst*. If *dst* is not provided the the destination file
        name is the MDF name. If overwrite is *True* then the destination file
        is overwritten, otherwise the file name is appended with '_<cntr>',
        were '<cntr>' is the first counter that produces a new file name (that
        does not already exist in the filesystem).

        Parameters
        ----------
        dst : str
            destination file name, Default ''
        overwrite : bool
            overwrite flag, default *False*
        compression : int
            does nothing for mdf version3; introduced here to share the same
            API as mdf version 4 files

        """
        # pylint: disable=unused-argument

        if self.file_history is None:
            self.file_history = TextBlock(text='''<FHcomment>
<TX>created</TX>
<tool_id>asammdf</tool_id>
<tool_vendor> </tool_vendor>
<tool_version>{}</tool_version>
</FHcomment>'''.format(__version__))
        else:
            text = '{}\n{}: updated by asammdf {}'
            old_history = self.file_history['text'].decode('latin-1')
            timestamp = time.asctime().encode('latin-1')

            text = text.format(
                old_history,
                timestamp,
                __version__,
            )
            self.file_history = TextBlock(text=text)

        if self.name is None and dst == '':
            message = (
                'Must specify a destination file name '
                'for MDF created from scratch'
            )
            raise MdfException(message)

        dst = dst if dst else self.name
        if not dst.endswith(('mdf', 'MDF')):
            dst = dst + '.mdf'
        if overwrite is False:
            if os.path.isfile(dst):
                cntr = 0
                while True:
                    name = os.path.splitext(dst)[0] + '_{}.mdf'.format(cntr)
                    if not os.path.isfile(name):
                        break
                    else:
                        cntr += 1
                message = (
                    'Destination file "{}" already exists '
                    'and "overwrite" is False. Saving MDF file as "{}"'
                )
                message = message.format(dst, name)
                logger.warning(message)
                dst = name

        if dst == self.name:
            destination = dst + '.temp'
        else:
            destination = dst

        with open(destination, 'wb+') as dst_:
            defined_texts = {}

            groups_nr = len(self.groups)

            write = dst_.write
            tell = dst_.tell
            seek = dst_.seek
            # list of all blocks
            blocks = []

            address = 0

            write(bytes(self.identification))

            write(bytes(self.header))

            if self.header.program:
                address = tell()
                write(bytes(self.header.program))
                self.header['program_addr'] = address
            else:
                self.header['program_addr'] = 0

            address = tell()
            self.file_history.address = address
            write(bytes(self.file_history))

            # DataGroup
            # put them first in the block list so they will be written first to
            # disk this way, in case of memory=False, we can safely
            # restore he original data block address

            data_address = []

            ce_map = {}

            for idx, gp in enumerate(self.groups):
                gp['temp_channels'] = ch_addrs = []
                gp['temp_channel_conversions'] = cc_addrs = []
                gp['temp_channel_extensions'] = ce_addrs = []
                gp['temp_channel_dependencies'] = cd_addrs = []

                gp_texts = deepcopy(gp['texts'])
                if gp['data_location'] == v23c.LOCATION_ORIGINAL_FILE:
                    stream = self._file
                else:
                    stream = self._tempfile

                # Texts
                for item_list in gp_texts.values():
                    for my_dict in item_list:
                        if my_dict is None:
                            continue
                        for key, tx_block in my_dict.items():

                            # text blocks can be shared
                            block = TextBlock(
                                address=tx_block,
                                stream=stream,
                            )
                            text = block['text']
                            if text in defined_texts:
                                my_dict[key] = defined_texts[text]
                            else:
                                address = tell()
                                defined_texts[text] = address
                                my_dict[key] = address
                                write(bytes(block))

                # Channel Dependency
                for dep in gp['channel_dependencies']:
                    if dep:
                        address = tell()
                        cd_addrs.append(address)
                        write(bytes(dep))
                    else:
                        cd_addrs.append(0)

                # channel extensions
                for addr in gp['channel_extensions']:
                    if addr:
                        stream.seek(addr)
                        raw_bytes = stream.read(v23c.CE_BLOCK_SIZE)
                        if raw_bytes in ce_map:
                            ce_addrs.append(ce_map[raw_bytes])
                        else:
                            address = tell()
                            source = ChannelExtension(raw_bytes=raw_bytes)
                            ce_map[raw_bytes] = address
                            ce_addrs.append(address)
                            write(bytes(source))
                    else:
                        ce_addrs.append(0)

                # ChannelConversions
                for i, addr in enumerate(gp['channel_conversions']):
                    if addr:
                        conversion = ChannelConversion(
                            address=addr,
                            stream=stream,
                        )

                        write_cc(conversion, defined_texts, stream=dst_)
                        address = tell()
                        cc_addrs.append(address)
                        write(bytes(conversion))
                    else:
                        cc_addrs.append(0)

                blocks = []

                for i, channel in enumerate(gp['channels']):
                    channel = Channel(
                        address=channel,
                        stream=stream,
                    )
                    blocks.append(channel)

                    if channel['comment_addr']:
                        tx_block = TextBlock(
                            address=channel['comment_addr'],
                            stream=stream,
                        )
                        text = tx_block['text']
                        if text in defined_texts:
                            channel['comment_addr'] = defined_texts[text].address
                        else:
                            address = tell()
                            channel['comment_addr'] = address
                            defined_texts[text] = tx_block
                            tx_block.address = address
                            write(bytes(tx_block))

                    if channel.get('long_name_addr', 0):
                        tx_block = TextBlock(
                            address=channel['long_name_addr'],
                            stream=stream,
                        )
                        text = tx_block['text']
                        if text in defined_texts:
                            channel['long_name_addr'] = defined_texts[text].address
                        else:
                            address = tell()
                            channel['long_name_addr'] = address
                            defined_texts[text] = tx_block
                            tx_block.address = address
                            write(bytes(tx_block))

                    if channel.get('display_name_addr', 0):
                        tx_block = TextBlock(
                            address=channel['display_name_addr'],
                            stream=stream,
                        )
                        text = tx_block['text']
                        if text in defined_texts:
                            channel['display_name_addr'] = defined_texts[text].address
                        else:
                            address = tell()
                            channel['display_name_addr'] = address
                            defined_texts[text] = tx_block
                            tx_block.address = address
                            write(bytes(tx_block))

                    channel['conversion_addr'] = cc_addrs[i]
                    channel['source_depend_addr'] = ce_addrs[i]
                    channel['ch_depend_addr'] = cd_addrs[i]

                address = tell()
                for block in blocks:
                    ch_addrs.append(address)
                    block.address = address
                    address += block['block_len']
                for j, block in enumerate(blocks[:-1]):
                    block['next_ch_addr'] = blocks[j + 1].address
                blocks[-1]['next_ch_addr'] = 0

                for block in blocks:
                    write(bytes(block))

                blocks = None

                address = tell()

                # ChannelGroup
                cg = gp['channel_group']
                cg.address = address

                cg['next_cg_addr'] = 0
                cg['first_ch_addr'] = ch_addrs[0]
                cg_texts = gp_texts['channel_group'][0]
                if 'comment_addr' in cg_texts:
                    addr = cg_texts['comment_addr']
                    cg['comment_addr'] = addr
                write(bytes(cg))

                address = tell()

                # TriggerBLock
                trigger, trigger_text = gp['trigger']
                if trigger:
                    if trigger_text:
                        trigger_text.address = address
                        write(bytes(trigger_text))
                        trigger['text_addr'] = trigger_text.address
                    else:
                        trigger['text_addr'] = 0

                    address = tell()
                    trigger.address = address
                    write(bytes(trigger))

                address = tell()

                # DataBlock
                data = self._load_group_data(gp)

                dat_addr = tell()
                for (data_bytes, _) in data:
                    write(data_bytes)

                if tell() - dat_addr:
                    data_address.append(dat_addr)
                else:
                    data_address.append(0)

                del gp['temp_channel_conversions']
                del gp['temp_channel_extensions']

                if self._callback:
                    self._callback(int(100 * (idx+1) / groups_nr), 100)

                if self._terminate:
                    dst_.close()
                    self.close()
                    return

            orig_addr = [
                gp['data_group']['data_block_addr']
                for gp in self.groups
            ]
            address = tell()
            gp_rec_ids = []
            for i, gp in enumerate(self.groups):
                dg = gp['data_group']
                gp_rec_ids.append(dg['record_id_nr'])
                dg['record_id_nr'] = 0
                dg['data_block_addr'] = data_address[i]
                dg.address = address
                address += dg['block_len']
                gp['data_group']['first_cg_addr'] = gp['channel_group'].address
                if gp['trigger'][0]:
                    gp['data_group']['trigger_addr'] = gp['trigger'][0].address
                else:
                    gp['data_group']['trigger_addr'] = 0

            if self.groups:
                for i, gp in enumerate(self.groups[:-1]):
                    addr = self.groups[i + 1]['data_group'].address
                    gp['data_group']['next_dg_addr'] = addr
                self.groups[-1]['data_group']['next_dg_addr'] = 0

            for i, gp in enumerate(self.groups):
                write(bytes(gp['data_group']))
                gp['data_block_addr'] = orig_addr[i]

            for gp, rec_id in zip(self.groups, gp_rec_ids):
                gp['data_group']['record_id_nr'] = rec_id

            if self.groups:
                address = self.groups[0]['data_group'].address
                self.header['first_dg_addr'] = address
                self.header['dg_nr'] = len(self.groups)
                self.header['comment_addr'] = self.file_history.address

            # update referenced channels addresses in the channel dependecies
            for gp in self.groups:
                for dep in gp['channel_dependencies']:
                    if not dep:
                        continue

                    for i, pair_ in enumerate(dep.referenced_channels):
                        _, dg_nr = pair_
                        grp = self.groups[dg_nr]
                        dep['ch_{}'.format(i)] = grp['temp_channels'][i]
                        dep['cg_{}'.format(i)] = grp['channel_group'].address
                        dep['dg_{}'.format(i)] = grp['data_group'].address
                    seek(dep.address)
                    write(bytes(dep))

            seek(v23c.ID_BLOCK_SIZE)
            write(bytes(self.header))

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

            self._master_channel_cache = {}

            self._tempfile = TemporaryFile()
            self._file = open(self.name, 'rb')
            self._read()
        return dst


if __name__ == '__main__':
    pass

# -*- coding: utf-8 -*-
""" classes that implement the blocks for MDF versions 2 and 3 """

from __future__ import division, print_function
import logging
import sys
import time
from datetime import datetime
from getpass import getuser
from struct import pack, unpack, unpack_from
from textwrap import wrap

import numpy as np
from numexpr import evaluate

from . import v2_v3_constants as v23c
from .utils import MdfException, get_text_v3, SignalSource
from .version import __version__

PYVERSION = sys.version_info[0]
SEEK_START = v23c.SEEK_START
SEEK_END = v23c.SEEK_END

if PYVERSION < 3:
    from .utils import bytes

logger = logging.getLogger('asammdf')


__all__ = [
    'Channel',
    'ChannelConversion',
    'ChannelDependency',
    'ChannelExtension',
    'ChannelGroup',
    'DataBlock',
    'DataGroup',
    'FileIdentificationBlock',
    'HeaderBlock',
    'ProgramBlock',
    'SampleReduction',
    'TextBlock',
    'TriggerBlock',
]


class Channel(dict):
    ''' CNBLOCK class

    If the `load_metadata` keyword argument is not provided or is False,
    then the conversion, source and display name information is not processed.

    *Channel* has the following key-value pairs

    * ``id`` - bytes : block ID; always b'CN'
    * ``block_len`` - int : block bytes size
    * ``next_ch_addr`` - int : next CNBLOCK address
    * ``conversion_addr`` - int : address of channel conversion block
    * ``source_addr`` - int : address of channel source block
    * ``ch_depend_addr`` - int : address of dependency block (CDBLOCK) of this
      channel
    * ``comment_addr`` - int : address of TXBLOCK that contains the
      channel comment
    * ``channel_type`` - int : integer code for channel type
    * ``short_name`` - bytes : short signal name
    * ``description`` - bytes : signal description
    * ``start_offset`` - int : start offset in bits to determine the first bit
      of the signal in the data record
    * ``bit_count`` - int : channel bit count
    * ``data_type`` - int : integer code for channel data type
    * ``range_flag`` - int : value range valid flag
    * ``min_raw_value`` - float : min raw value of all samples
    * ``max_raw_value`` - float : max raw value of all samples
    * ``sampling_rate`` - float : sampling rate in *'s'* for a virtual time
      channel
    * ``long_name_addr`` - int : address of TXBLOCK that contains the channel's
      name
    * ``display_name_addr`` - int : address of TXBLOCK that contains the
      channel's display name
    * ``aditional_byte_offset`` - int : additional Byte offset of the channel
      in the data recor

    Parameters
    ----------
    address : int
        block address; to be used for objects created from file
    stream : handle
        file handle; to be used for objects created from file
    load_metadata : bool
        option to load conversion, source and display_name; default *True*
    for dynamically created objects :
        see the key-value pairs

    Attributes
    ----------
    address : int
        block address inside mdf file
    comment : str
        channel comment
    conversion : ChannelConversion
        channel conversion; *None* if the channel has no conversion
    display_name : str
        channel display name
    name : str
        full channel name
    source : SourceInformation
        channel source information; *None* if the channel has no source
        information

    Examples
    --------
    >>> with open('test.mdf', 'rb') as mdf:
    ...     ch1 = Channel(stream=mdf, address=0xBA52)
    >>> ch2 = Channel()
    >>> ch1.name
    'VehicleSpeed'
    >>> ch1['id']
    b'CN'

    '''

    def __init__(self, **kwargs):
        super(Channel, self).__init__()

        self.name = self.display_name = self.comment = ''
        self.conversion = self.source = None

        try:
            stream = kwargs['stream']
            self.address = address = kwargs['address']
            stream.seek(address + 2)
            size = unpack('<H', stream.read(2))[0]
            stream.seek(address)
            block = stream.read(size)

            load_metadata = kwargs.get('load_metadata', True)

            if size == v23c.CN_DISPLAYNAME_BLOCK_SIZE:
                (self['id'],
                 self['block_len'],
                 self['next_ch_addr'],
                 self['conversion_addr'],
                 self['source_addr'],
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
                 self['aditional_byte_offset']) = unpack(
                    v23c.FMT_CHANNEL_DISPLAYNAME,
                    block,
                )

                addr = self['long_name_addr']
                if addr:
                    self.name = get_text_v3(
                        address=addr,
                        stream=stream,
                    )
                else:
                    self.name = (
                        self['short_name']
                        .decode('latin-1')
                        .strip(' \t\n\r\0')
                    )

                addr = self['display_name_addr']
                if addr:
                    self.display_name = get_text_v3(
                        address=addr,
                        stream=stream,
                    )

                if load_metadata:
                    addr = self['conversion_addr']
                    if addr:
                        self.conversion = ChannelConversion(
                            address=addr,
                            stream=stream,
                        )

                    addr = self['source_addr']
                    if addr:
                        self.source = ChannelExtension(
                            address=addr,
                            stream=stream,
                        )

                    addr = self['comment_addr']
                    if addr:
                        self.comment = get_text_v3(
                            address=addr,
                            stream=stream,
                        )

            elif size == v23c.CN_LONGNAME_BLOCK_SIZE:
                (self['id'],
                 self['block_len'],
                 self['next_ch_addr'],
                 self['conversion_addr'],
                 self['source_addr'],
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
                 self['long_name_addr']) = unpack(
                    v23c.FMT_CHANNEL_LONGNAME,
                    block,
                )

                addr = self['long_name_addr']
                if addr:
                    self.name = get_text_v3(
                        address=addr,
                        stream=stream,
                    )
                else:
                    self.name = (
                        self['short_name']
                        .decode('latin-1')
                        .strip(' \t\n\r\0')
                    )

                if load_metadata:

                    addr = self['conversion_addr']
                    if addr:
                        self.conversion = ChannelConversion(
                            address=addr,
                            stream=stream,
                        )

                    addr = self['source_addr']
                    if addr:
                        self.source = ChannelExtension(
                            address=addr,
                            stream=stream,
                        )

                    addr = self['comment_addr']
                    if addr:
                        self.comment = get_text_v3(
                            address=addr,
                            stream=stream,
                        )
            else:
                (self['id'],
                 self['block_len'],
                 self['next_ch_addr'],
                 self['conversion_addr'],
                 self['source_addr'],
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
                 self['sampling_rate']) = unpack(v23c.FMT_CHANNEL_SHORT, block)

                self.name = (
                    self['short_name']
                    .decode('latin-1')
                    .strip(' \t\n\r\0')
                )

                if load_metadata:

                    addr = self['conversion_addr']
                    if addr:
                        self.conversion = ChannelConversion(
                            address=addr,
                            stream=stream,
                        )

                    addr = self['source_addr']
                    if addr:
                        self.source = ChannelExtension(
                            address=addr,
                            stream=stream,
                        )

                    addr = self['comment_addr']
                    if addr:
                        self.comment = get_text_v3(
                            address=addr,
                            stream=stream,
                        )

            if self['id'] != b'CN':
                message = 'Expected "CN" block @{} but found "{}"'
                message = message.format(hex(address), self['id'])
                logger.exception(message)
                raise MdfException(message)

        except KeyError:

            self.address = 0
            self['id'] = b'CN'
            self['block_len'] = kwargs.get(
                'block_len',
                v23c.CN_DISPLAYNAME_BLOCK_SIZE,
            )
            self['next_ch_addr'] = kwargs.get('next_ch_addr', 0)
            self['conversion_addr'] = kwargs.get('conversion_addr', 0)
            self['source_addr'] = kwargs.get('source_addr', 0)
            self['ch_depend_addr'] = kwargs.get('ch_depend_addr', 0)
            self['comment_addr'] = kwargs.get('comment_addr', 0)
            self['channel_type'] = kwargs.get('channel_type', 0)
            self['short_name'] = kwargs.get('short_name', (b'\0' * 32))
            self['description'] = kwargs.get('description', (b'\0' * 128))
            self['start_offset'] = kwargs.get('start_offset', 0)
            self['bit_count'] = kwargs.get('bit_count', 8)
            self['data_type'] = kwargs.get('data_type', 0)
            self['range_flag'] = kwargs.get('range_flag', 1)
            self['min_raw_value'] = kwargs.get('min_raw_value', 0)
            self['max_raw_value'] = kwargs.get('max_raw_value', 0)
            self['sampling_rate'] = kwargs.get('sampling_rate', 0)
            if self['block_len'] >= v23c.CN_LONGNAME_BLOCK_SIZE:
                self['long_name_addr'] = kwargs.get('long_name_addr', 0)
            if self['block_len'] >= v23c.CN_DISPLAYNAME_BLOCK_SIZE:
                self['display_name_addr'] = kwargs.get('display_name_addr', 0)
                self['aditional_byte_offset'] = kwargs.get(
                    'aditional_byte_offset',
                    0,
                )

    def to_blocks(self, address, blocks, defined_texts, cc_map, si_map):
        key = 'long_name_addr'
        text = self.name
        if key in self:
            if len(text) > 31:
                if text in defined_texts:
                    self[key] = defined_texts[text]
                else:
                    tx_block = TextBlock(text=text)
                    self[key] = address
                    defined_texts[text] = address
                    tx_block.address = address
                    address += tx_block['block_len']
                    blocks.append(tx_block)
            else:
                self[key] = 0

        self['short_name'] = text.encode('latin-1')[:31]

        key = 'display_name_addr'
        text = self.display_name
        if key in self:
            if text:
                if text in defined_texts:
                    self[key] = defined_texts[text]
                else:
                    tx_block = TextBlock(text=text)
                    self[key] = address
                    defined_texts[text] = address
                    tx_block.address = address
                    address += tx_block['block_len']
                    blocks.append(tx_block)
            else:
                self[key] = 0

        key = 'comment_addr'
        text = self.comment
        if text:
            if len(text) < 128:
                self['description'] = text.encode('latin-1')[:127]
                self[key] = 0
            else:
                if text in defined_texts:
                    self[key] = defined_texts[text]
                else:
                    tx_block = TextBlock(text=text)
                    self[key] = address
                    defined_texts[text] = address
                    tx_block.address = address
                    address += tx_block['block_len']
                    blocks.append(tx_block)
                self['description'] = b'\0'
        else:
            self[key] = 0

        conversion = self.conversion
        if conversion:
            address = conversion.to_blocks(address, blocks, defined_texts, cc_map)
            self['conversion_addr'] = conversion.address
        else:
            self['conversion_addr'] = 0

        source = self.source
        if source:
            address = source.to_blocks(address, blocks, defined_texts, si_map)
            self['source_addr'] = source.address
        else:
            self['source_addr'] = 0

        blocks.append(self)
        self.address = address
        address += self['block_len']

        return address

    def to_stream(self, stream, defined_texts, cc_map, si_map):
        address = stream.tell()

        key = 'long_name_addr'
        text = self.name
        if key in self:
            if len(text) > 31:
                if text in defined_texts:
                    self[key] = defined_texts[text]
                else:
                    tx_block = TextBlock(text=text)
                    self[key] = address
                    defined_texts[text] = address
                    tx_block.address = address
                    address += tx_block['block_len']
                    stream.write(bytes(tx_block))
            else:
                self[key] = 0
        self['short_name'] = text.encode('latin-1')[:31]

        key = 'display_name_addr'
        text = self.display_name
        if key in self:
            if text:
                if text in defined_texts:
                    self[key] = defined_texts[text]
                else:
                    tx_block = TextBlock(text=text)
                    self[key] = address
                    defined_texts[text] = address
                    tx_block.address = address
                    address += tx_block['block_len']
                    stream.write(bytes(tx_block))
            else:
                self[key] = 0

        key = 'comment_addr'
        text = self.comment
        if text:
            if len(text) < 128:
                self['description'] = text.encode('latin-1')[:127]
                self[key] = 0
            else:
                if text in defined_texts:
                    self[key] = defined_texts[text]
                else:
                    tx_block = TextBlock(text=text)
                    self[key] = address
                    defined_texts[text] = address
                    tx_block.address = address
                    address += tx_block['block_len']
                    stream.write(bytes(tx_block))
                self['description'] = b'\0'
        else:
            self[key] = 0

        conversion = self.conversion
        if conversion:
            address = conversion.to_stream(stream, defined_texts, cc_map)
            self['conversion_addr'] = conversion.address
        else:
            self['conversion_addr'] = 0

        source = self.source
        if source:
            address = source.to_stream(stream, defined_texts, si_map)
            self['source_addr'] = source.address
        else:
            self['source_addr'] = 0

        stream.write(bytes(self))
        self.address = address
        address += self['block_len']

        return address

    def metadata(self):
        max_len = max(
            len(key)
            for key in self
        )
        template = '{{: <{}}}: {{}}'.format(max_len)

        metadata = []
        lines = """
name: {}
display name: {}
address: {}
comment: {}

""".format(
            self.name,
            self.display_name,
            hex(self.address),
            self.comment,
        ).split('\n')
        for key, val in self.items():
            if key.endswith('addr') or key.startswith('text_'):
                lines.append(
                    template.format(key, hex(val))
                )
            elif isinstance(val, float):
                    lines.append(
                        template.format(key, round(val, 6))
                    )
            else:
                if (PYVERSION < 3 and isinstance(val, str)) or \
                        (PYVERSION >= 3 and isinstance(val, bytes)):
                    lines.append(
                        template.format(key, val.strip(b'\0'))
                    )
                else:
                    lines.append(
                        template.format(key, val)
                    )
        for line in lines:
            if not line:
                metadata.append(line)
            else:
                for wrapped_line in wrap(line, width=120):
                    metadata.append(wrapped_line)

        return '\n'.join(metadata)

    def __bytes__(self):

        block_len = self['block_len']
        if block_len == v23c.CN_DISPLAYNAME_BLOCK_SIZE:
            fmt = v23c.FMT_CHANNEL_DISPLAYNAME
            keys = v23c.KEYS_CHANNEL_DISPLAYNAME
        elif block_len == v23c.CN_LONGNAME_BLOCK_SIZE:
            fmt = v23c.FMT_CHANNEL_LONGNAME
            keys = v23c.KEYS_CHANNEL_LONGNAME
        else:
            fmt = v23c.FMT_CHANNEL_SHORT
            keys = v23c.KEYS_CHANNEL_SHORT

        result = pack(fmt, *[self[key] for key in keys])
        return result

    def __lt__(self, other):
        self_start = self['start_offset']
        other_start = other['start_offset']
        try:
            self_additional_offset = self['aditional_byte_offset']
            if self_additional_offset:
                self_start += 8 * self_additional_offset
            other_additional_offset = other['aditional_byte_offset']
            if other_additional_offset:
                other_start += 8 * other_additional_offset
        except KeyError:
            pass

        if self_start < other_start:
            result = 1
        elif self_start == other_start:
            if self['bit_count'] >= other['bit_count']:
                result = 1
            else:
                result = 0
        else:
            result = 0
        return result

    def __repr__(self):
        return 'Channel (name: {}, display name: {}, comment: {}, address: {}, fields: {})'.format(
            self.name,
            self.display_name,
            self.comment,
            hex(self.address),
            dict(self),
        )


class ChannelConversion(dict):
    ''' CCBLOCK class

    The ChannelConversion object can be created in two modes:

    *ChannelConversion* has the following common key-value pairs

    * ``id`` - bytes : block ID; always b'CC'
    * ``block_len`` - int : block bytes size
    * ``range_flag`` - int : value range valid flag
    * ``min_phy_value`` - float : min raw value of all samples
    * ``max_phy_value`` - float : max raw value of all samples
    * ``unit`` - bytes : physical unit
    * ``conversion_type`` - int : integer code for conversion type
    * ``ref_param_nr`` - int : number of referenced parameters

    *ChannelConversion* has the following specific key-value pairs

    * linear conversion

        * ``a`` - float : factor
        * ``b`` - float : offset
        * ``CANapeHiddenExtra`` - bytes : sometimes CANape appends extra
          information; not compliant with MDF specs

    * algebraic conversion

        * ``formula`` - bytes : ecuation as string

    * polynomial or rational conversion

        * ``P1`` to ``P6`` - float : parameters

    * exponential or logarithmic conversion

        * ``P1`` to ``P7`` - float : parameters

    * tabular with or without interpolation (grouped by index)

        * ``raw_<N>`` - int : N-th raw value (X axis)
        * ``phys_<N>`` - float : N-th physical value (Y axis)

    * text table conversion

        * ``param_val_<N>`` - int : N-th raw value (X axis)
        * ``text_<N>`` - N-th text physical value (Y axis)

    * text range table conversion

        * ``default_lower`` - float : default lower raw value
        * ``default_upper`` - float : default upper raw value
        * ``default_addr`` - int : address of default text physical value
        * ``lower_<N>`` - float : N-th lower raw value
        * ``upper_<N>`` - float : N-th upper raw value
        * ``text_<N>`` - int : address of N-th text physical value

    Parameters
    ----------
    address : int
        block address inside mdf file
    raw_bytes : bytes
        complete block read from disk
    stream : file handle
        mdf file handle
    for dynamically created objects :
        see the key-value pairs

    Attributes
    ----------
    address : int
        block address inside mdf file
    formula : str
        formula string in case of algebraic conversion
    referenced_blocks : list
        list of CCBLOCK/TXBLOCK referenced by the conversion
    unit : str
        physical unit

    Examples
    --------
    >>> with open('test.mdf', 'rb') as mdf:
    ...     cc1 = ChannelConversion(stream=mdf, address=0xBA52)
    >>> cc2 = ChannelConversion(conversion_type=0)
    >>> cc1['b'], cc1['a']
    0, 100.0

    '''

    def __init__(self, **kwargs):
        super(ChannelConversion, self).__init__()

        self.unit = self.formula = ''

        self.referenced_blocks = {}

        if 'raw_bytes' in kwargs or 'stream' in kwargs:
            try:
                self.address = 0
                block = kwargs['raw_bytes']
                (self['id'],
                 self['block_len']) = unpack_from(
                    '<2sH',
                    block,
                )
                size = self['block_len']
                block_size = len(block)
                block = block[4:]
                stream = kwargs['stream']

            except KeyError:
                stream = kwargs['stream']
                self.address = address = kwargs['address']
                stream.seek(address)
                block = stream.read(4)
                (self['id'],
                 self['block_len']) = unpack('<2sH', block)

                size = self['block_len']
                block_size = size
                block = stream.read(size - 4)

            address = kwargs.get('address', 0)
            self.address = address

            (self['range_flag'],
             self['min_phy_value'],
             self['max_phy_value'],
             self['unit'],
             self['conversion_type'],
             self['ref_param_nr']) = unpack_from(
                v23c.FMT_CONVERSION_COMMON_SHORT,
                block,
            )

            self.unit = self['unit'].decode('latin-1').strip(' \t\r\n\0')

            conv_type = self['conversion_type']

            if conv_type == v23c.CONVERSION_TYPE_LINEAR:
                (self['b'],
                 self['a']) = unpack_from(
                    '<2d',
                    block,
                    v23c.CC_COMMON_SHORT_SIZE,
                )
                if not size == v23c.CC_LIN_BLOCK_SIZE:
                    self['CANapeHiddenExtra'] = block[v23c.CC_LIN_BLOCK_SIZE - 4:]

            elif conv_type == v23c.CONVERSION_TYPE_NONE:
                pass

            elif conv_type == v23c.CONVERSION_TYPE_FORMULA:
                self['formula'] = block[v23c.CC_COMMON_SHORT_SIZE:]
                self.formula = self['formula'].decode('latin-1').strip(' \t\r\n\0')

            elif conv_type in (
                    v23c.CONVERSION_TYPE_TABI,
                    v23c.CONVERSION_TYPE_TAB):

                nr = self['ref_param_nr']

                size = v23c.CC_COMMON_BLOCK_SIZE + nr * 16

                if block_size == v23c.MAX_UINT16:
                    stream.seek(address)
                    raw_bytes = stream.read(size)
                    conversion = ChannelConversion(
                        raw_bytes=raw_bytes,
                        stream=stream,
                        address=address,
                    )
                    conversion['block_len'] = size

                    self.update(conversion)
                    self.referenced_blocks = conversion.referenced_blocks

                else:
                    values = unpack_from(
                        '<{}d'.format(2 * nr),
                        block,
                        v23c.CC_COMMON_SHORT_SIZE,
                    )
                    for i in range(nr):
                        (self['raw_{}'.format(i)],
                         self['phys_{}'.format(i)]) = values[i*2], values[2*i + 1]

            elif conv_type in (
                    v23c.CONVERSION_TYPE_POLY,
                    v23c.CONVERSION_TYPE_RAT):
                (self['P1'],
                 self['P2'],
                 self['P3'],
                 self['P4'],
                 self['P5'],
                 self['P6']) = unpack_from(
                    '<6d',
                    block,
                    v23c.CC_COMMON_SHORT_SIZE,
                )

            elif conv_type in (
                    v23c.CONVERSION_TYPE_EXPO,
                    v23c.CONVERSION_TYPE_LOGH):
                (self['P1'],
                 self['P2'],
                 self['P3'],
                 self['P4'],
                 self['P5'],
                 self['P6'],
                 self['P7']) = unpack_from(
                    '<7d',
                    block,
                    v23c.CC_COMMON_SHORT_SIZE,
                )

            elif conv_type == v23c.CONVERSION_TYPE_TABX:
                nr = self['ref_param_nr']

                size = v23c.CC_COMMON_BLOCK_SIZE + nr * 40

                if block_size == v23c.MAX_UINT16:
                    stream.seek(address)
                    raw_bytes = stream.read(size)
                    conversion = ChannelConversion(
                        raw_bytes=raw_bytes,
                        stream=stream,
                        address=address,
                    )
                    conversion['block_len'] = size

                    self.update(conversion)
                    self.referenced_blocks = conversion.referenced_blocks

                else:

                    values = unpack_from(
                        '<' + 'd32s' * nr,
                        block,
                        v23c.CC_COMMON_SHORT_SIZE,
                    )

                    for i in range(nr):
                        (self['param_val_{}'.format(i)],
                         self['text_{}'.format(i)]) = values[i*2], values[2*i + 1]

            elif conv_type == v23c.CONVERSION_TYPE_RTABX:

                nr = self['ref_param_nr'] - 1

                size = v23c.CC_COMMON_BLOCK_SIZE + (nr + 1) * 20

                if block_size == v23c.MAX_UINT16:
                    stream.seek(address)
                    raw_bytes = stream.read(size)
                    conversion = ChannelConversion(
                        raw_bytes=raw_bytes,
                        stream=stream,
                        address=address,
                    )
                    conversion['block_len'] = size

                    self.update(conversion)
                    self.referenced_blocks = conversion.referenced_blocks

                else:

                    (self['default_lower'],
                     self['default_upper'],
                     self['default_addr']) = unpack_from(
                        '<2dI',
                        block,
                        v23c.CC_COMMON_SHORT_SIZE,
                    )

                    if self['default_addr']:
                        self.referenced_blocks['default_addr'] = TextBlock(
                            address=self['default_addr'],
                            stream=stream,
                        )
                    else:
                        self.referenced_blocks['default_addr'] = TextBlock(
                            text='',
                        )

                    values = unpack_from(
                        '<' + '2dI' * nr,
                        block,
                        v23c.CC_COMMON_SHORT_SIZE + 20,
                    )
                    for i in range(nr):
                        (self['lower_{}'.format(i)],
                         self['upper_{}'.format(i)],
                         self['text_{}'.format(i)]) = (
                            values[i*3],
                            values[3*i + 1],
                            values[3*i + 2],
                        )
                        if values[3*i + 2]:
                            block = TextBlock(
                                address=values[3*i + 2],
                                stream=stream,
                            )
                            self.referenced_blocks['text_{}'.format(i)] = block

                        else:
                            self.referenced_blocks['text_{}'.format(i)] = TextBlock(
                                text='',
                            )

            if self['id'] != b'CC':
                message = 'Expected "CC" block @{} but found "{}"'
                message = message.format(hex(address), self['id'])
                logger.exception(message)
                raise MdfException(message)

        else:

            self.address = 0
            self['id'] = 'CC'.encode('latin-1')

            if kwargs['conversion_type'] == v23c.CONVERSION_TYPE_NONE:
                self['block_len'] = v23c.CC_COMMON_BLOCK_SIZE
                self['range_flag'] = kwargs.get('range_flag', 1)
                self['min_phy_value'] = kwargs.get('min_phy_value', 0)
                self['max_phy_value'] = kwargs.get('max_phy_value', 0)
                self['unit'] = kwargs.get('unit', ('\0' * 20).encode('latin-1'))
                self['conversion_type'] = v23c.CONVERSION_TYPE_NONE
                self['ref_param_nr'] = 0

            elif kwargs['conversion_type'] == v23c.CONVERSION_TYPE_LINEAR:
                self['block_len'] = v23c.CC_LIN_BLOCK_SIZE
                self['range_flag'] = kwargs.get('range_flag', 1)
                self['min_phy_value'] = kwargs.get('min_phy_value', 0)
                self['max_phy_value'] = kwargs.get('max_phy_value', 0)
                self['unit'] = kwargs.get('unit', ('\0' * 20).encode('latin-1'))
                self['conversion_type'] = v23c.CONVERSION_TYPE_LINEAR
                self['ref_param_nr'] = 2
                self['b'] = kwargs.get('b', 0)
                self['a'] = kwargs.get('a', 1)
                if not self['block_len'] == v23c.CC_LIN_BLOCK_SIZE:
                    self['CANapeHiddenExtra'] = kwargs['CANapeHiddenExtra']

            elif kwargs['conversion_type'] in (
                    v23c.CONVERSION_TYPE_POLY,
                    v23c.CONVERSION_TYPE_RAT):
                self['block_len'] = v23c.CC_POLY_BLOCK_SIZE
                self['range_flag'] = kwargs.get('range_flag', 1)
                self['min_phy_value'] = kwargs.get('min_phy_value', 0)
                self['max_phy_value'] = kwargs.get('max_phy_value', 0)
                self['unit'] = kwargs.get('unit', ('\0' * 20).encode('latin-1'))
                self['conversion_type'] = kwargs['conversion_type']
                self['ref_param_nr'] = 6
                self['P1'] = kwargs.get('P1', 0)
                self['P2'] = kwargs.get('P2', 0)
                self['P3'] = kwargs.get('P3', 0)
                self['P4'] = kwargs.get('P4', 0)
                self['P5'] = kwargs.get('P5', 0)
                self['P6'] = kwargs.get('P6', 0)

            elif kwargs['conversion_type'] in (
                    v23c.CONVERSION_TYPE_EXPO,
                    v23c.CONVERSION_TYPE_LOGH):
                self['block_len'] = v23c.CC_EXPO_BLOCK_SIZE
                self['range_flag'] = kwargs.get('range_flag', 1)
                self['min_phy_value'] = kwargs.get('min_phy_value', 0)
                self['max_phy_value'] = kwargs.get('max_phy_value', 0)
                self['unit'] = kwargs.get('unit', ('\0' * 20).encode('latin-1'))
                self['conversion_type'] = v23c.CONVERSION_TYPE_EXPO
                self['ref_param_nr'] = 7
                self['P1'] = kwargs.get('P1', 0)
                self['P2'] = kwargs.get('P2', 0)
                self['P3'] = kwargs.get('P3', 0)
                self['P4'] = kwargs.get('P4', 0)
                self['P5'] = kwargs.get('P5', 0)
                self['P6'] = kwargs.get('P6', 0)
                self['P7'] = kwargs.get('P7', 0)

            elif kwargs['conversion_type'] == v23c.CONVERSION_TYPE_FORMULA:
                formula = kwargs['formula']
                formula_len = len(formula)
                try:
                    self.formula = formula.decode('latin-1')
                    formula += b'\0'
                except:
                    self.formula = formula
                    formula = formula.encode('latin-1') + b'\0'
                self['block_len'] = 46 + formula_len + 1
                self['range_flag'] = kwargs.get('range_flag', 1)
                self['min_phy_value'] = kwargs.get('min_phy_value', 0)
                self['max_phy_value'] = kwargs.get('max_phy_value', 0)
                self['unit'] = kwargs.get('unit', ('\0' * 20).encode('latin-1'))
                self['conversion_type'] = v23c.CONVERSION_TYPE_FORMULA
                self['ref_param_nr'] = formula_len
                self['formula'] = formula

            elif kwargs['conversion_type'] in (
                    v23c.CONVERSION_TYPE_TABI,
                    v23c.CONVERSION_TYPE_TAB):
                nr = kwargs['ref_param_nr']
                self['block_len'] = v23c.CC_COMMON_BLOCK_SIZE + nr * 2 * 8
                self['range_flag'] = kwargs.get('range_flag', 1)
                self['min_phy_value'] = kwargs.get('min_phy_value', 0)
                self['max_phy_value'] = kwargs.get('max_phy_value', 0)
                self['unit'] = kwargs.get('unit', ('\0' * 20).encode('latin-1'))
                self['conversion_type'] = kwargs['conversion_type']
                self['ref_param_nr'] = nr
                for i in range(nr):
                    self['raw_{}'.format(i)] = kwargs['raw_{}'.format(i)]
                    self['phys_{}'.format(i)] = kwargs['phys_{}'.format(i)]

            elif kwargs['conversion_type'] == v23c.CONVERSION_TYPE_TABX:
                nr = kwargs['ref_param_nr']
                self['block_len'] = v23c.CC_COMMON_BLOCK_SIZE + 40 * nr
                self['range_flag'] = kwargs.get('range_flag', 0)
                self['min_phy_value'] = kwargs.get('min_phy_value', 0)
                self['max_phy_value'] = kwargs.get('max_phy_value', 0)
                self['unit'] = kwargs.get('unit', ('\0' * 20).encode('latin-1'))
                self['conversion_type'] = v23c.CONVERSION_TYPE_TABX
                self['ref_param_nr'] = nr

                for i in range(nr):
                    self['param_val_{}'.format(i)] = kwargs['param_val_{}'.format(i)]
                    self['text_{}'.format(i)] = kwargs['text_{}'.format(i)]

            elif kwargs['conversion_type'] == v23c.CONVERSION_TYPE_RTABX:
                nr = kwargs['ref_param_nr']
                self['block_len'] = v23c.CC_COMMON_BLOCK_SIZE + 20 * nr
                self['range_flag'] = kwargs.get('range_flag', 0)
                self['min_phy_value'] = kwargs.get('min_phy_value', 0)
                self['max_phy_value'] = kwargs.get('max_phy_value', 0)
                self['unit'] = kwargs.get('unit', ('\0' * 20).encode('latin-1'))
                self['conversion_type'] = v23c.CONVERSION_TYPE_RTABX
                self['ref_param_nr'] = nr

                self['default_lower'] = 0
                self['default_upper'] = 0
                self['default_addr'] = 0
                key = 'default_addr'
                if key in kwargs:
                    self.referenced_blocks[key] = TextBlock(text=kwargs[key])
                else:
                    self.referenced_blocks[key] = None

                for i in range(nr - 1):
                    self['lower_{}'.format(i)] = kwargs['lower_{}'.format(i)]
                    self['upper_{}'.format(i)] = kwargs['upper_{}'.format(i)]
                    key = 'text_{}'.format(i)
                    self[key] = 0
                    self.referenced_blocks[key] = TextBlock(text=kwargs[key])
            else:
                message = 'Conversion type "{}" not implemented'
                message = message.format(kwargs['conversion_type'])
                logger.exception(message)
                raise MdfException(message)

    def to_blocks(self, address, blocks, defined_texts, cc_map):

        self['unit'] = self.unit.encode('latin-1')[:19]

        if 'formula' in self:
            formula = self.formula
            if not formula.endswith('\0'):
                formula += '\0'
            self['formula'] = formula.encode('latin-1')
            self['block_len'] = v23c.CC_COMMON_BLOCK_SIZE + len(self['formula'])

        for key, block in self.referenced_blocks.items():
            if block:
                if block['id'] == b'TX':
                    text = block['text']
                    if text in defined_texts:
                        self[key] = defined_texts[text]
                    else:
                        defined_texts[text] = address
                        blocks.append(block)
                        self[key] = address
                        address += block['block_len']
                else:
                    address = block.to_blocks(address, blocks, defined_texts, cc_map)
                    self[key] = block.address
            else:
                self[key] = 0

        bts = bytes(self)
        if bts in cc_map:
            self.address = cc_map[bts]
        else:
            blocks.append(bts)
            self.address = address
            cc_map[bts] = address
            address += self['block_len']

        return address

    def to_stream(self, stream, defined_texts, cc_map):
        address = stream.tell()

        self['unit'] = self.unit.encode('latin-1')[:19]

        if 'formula' in self:
            formula = self.formula
            if not formula.endswith('\0'):
                formula += '\0'
            self['formula'] = formula.encode('latin-1')
            self['block_len'] = v23c.CC_COMMON_BLOCK_SIZE + len(self['formula'])

        for key, block in self.referenced_blocks.items():
            if block:
                if block['id'] == b'TX':
                    text = block['text']
                    if text in defined_texts:
                        self[key] = defined_texts[text]
                    else:
                        defined_texts[text] = address
                        self[key] = address
                        address += block['block_len']
                        stream.write(bytes(block))
                else:
                    address = block.to_stream(stream, defined_texts, cc_map)
                    self[key] = block.address
            else:
                self[key] = 0

        bts = bytes(self)
        if bts in cc_map:
            self.address = cc_map[bts]
        else:
            cc_map[bts] = address
            stream.write(bytes(self))
            self.address = address
            address += self['block_len']

        return address

    def metadata(self, indent=''):
        max_len = max(
            len(key)
            for key in self
        )
        template = '{{: <{}}}: {{}}'.format(max_len)

        metadata = []
        lines = """
address: {}

""".format(
            hex(self.address),
        ).split('\n')
        for key, val in self.items():
            if key.endswith('addr') or key.startswith('text_'):
                lines.append(
                    template.format(key, hex(val))
                )
            elif isinstance(val, float):
                    lines.append(
                        template.format(key, round(val, 6))
                    )
            else:
                if (PYVERSION < 3 and isinstance(val, str)) or \
                        (PYVERSION >= 3 and isinstance(val, bytes)):
                    lines.append(
                        template.format(key, val.strip(b'\0'))
                    )
                else:
                    lines.append(
                        template.format(key, val)
                    )
        if self.referenced_blocks:
            max_len = max(
                len(key)
                for key in self.referenced_blocks
            )
            template = '{{: <{}}}: {{}}'.format(max_len)

            lines.append('')
            lines.append('Referenced blocks:')
            for key, block in self.referenced_blocks.items():
                if isinstance(block, TextBlock):
                    lines.append(
                        template.format(key, block['text'].strip(b'\0'))
                    )
                else:
                    lines.append(template.format(key, ''))
                    lines.extend(
                        block.metadata(indent + '    ').split('\n')
                    )

        for line in lines:
            if not line:
                metadata.append(line)
            else:
                for wrapped_line in wrap(
                        line,
                        initial_indent=indent,
                        subsequent_indent=indent,
                        width=120):
                    metadata.append(wrapped_line)

        return '\n'.join(metadata)

    def convert(self, values):
        numexpr_favorable_size = 140000
        conversion_type = self['conversion_type']

        if conversion_type == v23c.CONVERSION_TYPE_NONE:
            pass

        elif conversion_type == v23c.CONVERSION_TYPE_LINEAR:
            a = self['a']
            b = self['b']
            if (a, b) != (1, 0):
                if len(values) >= numexpr_favorable_size:
                    values = values.astype('float64')
                    values = evaluate("values * a + b")
                else:
                    values = values * a
                    if b:
                        values += b

        elif conversion_type in (
                v23c.CONVERSION_TYPE_TABI,
                v23c.CONVERSION_TYPE_TAB):
            nr = self['ref_param_nr']

            raw_vals = [
                self['raw_{}'.format(i)]
                for i in range(nr)
            ]
            raw_vals = np.array(raw_vals)
            phys = [
                self['phys_{}'.format(i)]
                for i in range(nr)
            ]
            phys = np.array(phys)

            if conversion_type == v23c.CONVERSION_TYPE_TABI:
                values = np.interp(values, raw_vals, phys)
            else:
                idx = np.searchsorted(raw_vals, values)
                idx = np.clip(idx, 0, len(raw_vals) - 1)
                values = phys[idx]

        elif conversion_type == v23c.CONVERSION_TYPE_TABX:
            nr = self['ref_param_nr']
            raw_vals = [
                self['param_val_{}'.format(i)]
                for i in range(nr)
            ]
            raw_vals = np.array(raw_vals)
            phys = [
                self['text_{}'.format(i)]
                for i in range(nr)
            ]
            phys = np.array(phys)

            indexes = np.searchsorted(raw_vals, values)

            values = phys[indexes]

        elif conversion_type == v23c.CONVERSION_TYPE_RTABX:
            nr = self['ref_param_nr'] - 1

            phys = []
            for i in range(nr):
                value = self.referenced_blocks['text_{}'.format(i)]
                if value:
                    value = value['text']
                else:
                    value = b''
                phys.append(value)

            phys = np.array(phys)

            default = self.referenced_blocks['default_addr']
            if default:
                default = default['text']
            else:
                default = b''

            if b'{X}' in default:
                default = (
                    default
                    .decode('latin-1')
                    .replace('{X}', 'X')
                    .split('"')
                    [1]
                )
                partial_conversion = True
            else:
                partial_conversion = False

            lower = np.array(
                [self['lower_{}'.format(i)] for i in range(nr)]
            )
            upper = np.array(
                [self['upper_{}'.format(i)] for i in range(nr)]
            )

            if values.dtype.kind == 'f':
                idx1 = np.searchsorted(lower, values, side='right') - 1
                idx2 = np.searchsorted(upper, values, side='right')
            else:
                idx1 = np.searchsorted(lower, values, side='right') - 1
                idx2 = np.searchsorted(upper, values, side='right') - 1

            idx = np.argwhere(idx1 != idx2).flatten()

            if partial_conversion and len(idx):
                X = values[idx]
                new_values = np.zeros(len(values), dtype=np.float64)
                new_values[idx] = evaluate(default)

                idx = np.argwhere(idx1 == idx2).flatten()
                new_values[idx] = np.nan
                values = new_values

            else:
                if len(idx):
                    new_values = np.zeros(
                        len(values),
                        dtype=max(phys.dtype, np.array([default, ]).dtype),
                    )
                    new_values[idx] = default

                    idx = np.argwhere(idx1 == idx2).flatten()
                    new_values[idx] = phys[values[idx]]
                    values = new_values
                else:
                    values = phys[idx1]

        elif conversion_type in (
                v23c.CONVERSION_TYPE_EXPO,
                v23c.CONVERSION_TYPE_LOGH):
            # pylint: disable=C0103

            if conversion_type == v23c.CONVERSION_TYPE_EXPO:
                func = np.log
            else:
                func = np.exp
            P1 = self['P1']
            P2 = self['P2']
            P3 = self['P3']
            P4 = self['P4']
            P5 = self['P5']
            P6 = self['P6']
            P7 = self['P7']
            if P4 == 0:
                values = func(((values - P7) * P6 - P3) / P1) / P2
            elif P1 == 0:
                values = func((P3 / (values - P7) - P6) / P4) / P5
            else:
                message = 'wrong conversion {}'
                message = message.format(conversion_type)
                raise ValueError(message)

        elif conversion_type == v23c.CONVERSION_TYPE_RAT:
            # pylint: disable=unused-variable,C0103

            P1 = self['P1']
            P2 = self['P2']
            P3 = self['P3']
            P4 = self['P4']
            P5 = self['P5']
            P6 = self['P6']

            X = values
            if (P1, P4, P5, P6) == (0, 0, 0, 1):
                if (P2, P3) != (1, 0):
                    if len(values) >= numexpr_favorable_size:
                        values = evaluate("values * P2 + P3")
                    else:
                        values = values * P2
                        if P3:
                            values += P3
            elif (P3, P4, P5, P6) == (0, 0, 1, 0):
                if (P1, P2) != (1, 0):
                    if len(values) >= numexpr_favorable_size:
                        values = evaluate("values * P1 + P2")
                    else:
                        values = values * P1
                        if P2:
                            values += P2
            else:
                values = evaluate(v23c.RAT_CONV_TEXT)

        elif conversion_type == v23c.CONVERSION_TYPE_POLY:
            # pylint: disable=unused-variable,C0103

            P1 = self['P1']
            P2 = self['P2']
            P3 = self['P3']
            P4 = self['P4']
            P5 = self['P5']
            P6 = self['P6']

            X = values

            coefs = (P2, P3, P5, P6)
            if coefs == (0, 0, 0, 0):
                if P1 != P4:
                    values = evaluate(v23c.POLY_CONV_SHORT_TEXT)
            else:
                values = evaluate(v23c.POLY_CONV_LONG_TEXT)

        elif conversion_type == v23c.CONVERSION_TYPE_FORMULA:
            # pylint: disable=unused-variable,C0103

            formula = self['formula'].decode('latin-1').strip(' \r\n\t\0')
            if 'X1' not in formula:
                formula = formula.replace('X', 'X1')
            X1 = values
            values = evaluate(formula)

        return values

    def __bytes__(self):
        conv = self['conversion_type']

        # compute the fmt
        if conv == v23c.CONVERSION_TYPE_NONE:
            fmt = v23c.FMT_CONVERSION_COMMON
        elif conv == v23c.CONVERSION_TYPE_FORMULA:
            fmt = v23c.FMT_CONVERSION_FORMULA.format(self['block_len'] - v23c.CC_COMMON_BLOCK_SIZE)
        elif conv == v23c.CONVERSION_TYPE_LINEAR:
            fmt = v23c.FMT_CONVERSION_LINEAR
            if not self['block_len'] == v23c.CC_LIN_BLOCK_SIZE:
                fmt += '{}s'.format(self['block_len'] - v23c.CC_LIN_BLOCK_SIZE)
        elif conv in (v23c.CONVERSION_TYPE_POLY, v23c.CONVERSION_TYPE_RAT):
            fmt = v23c.FMT_CONVERSION_POLY_RAT
        elif conv in (v23c.CONVERSION_TYPE_EXPO, v23c.CONVERSION_TYPE_LOGH):
            fmt = v23c.FMT_CONVERSION_EXPO_LOGH
        elif conv in (v23c.CONVERSION_TYPE_TABI, v23c.CONVERSION_TYPE_TAB):
            nr = self['ref_param_nr']
            fmt = v23c.FMT_CONVERSION_COMMON + '{}d'.format(nr * 2)
        elif conv == v23c.CONVERSION_TYPE_RTABX:
            nr = self['ref_param_nr']
            fmt = v23c.FMT_CONVERSION_COMMON + '2dI' * nr
        elif conv == v23c.CONVERSION_TYPE_TABX:
            nr = self['ref_param_nr']
            fmt = v23c.FMT_CONVERSION_COMMON + 'd32s' * nr

        if conv == v23c.CONVERSION_TYPE_NONE:
            keys = v23c.KEYS_CONVESION_NONE
        elif conv == v23c.CONVERSION_TYPE_FORMULA:
            keys = v23c.KEYS_CONVESION_FORMULA
        elif conv == v23c.CONVERSION_TYPE_LINEAR:
            keys = v23c.KEYS_CONVESION_LINEAR
            if not self['block_len'] == v23c.CC_LIN_BLOCK_SIZE:
                keys += ('CANapeHiddenExtra',)
        elif conv in (v23c.CONVERSION_TYPE_POLY, v23c.CONVERSION_TYPE_RAT):
            keys = v23c.KEYS_CONVESION_POLY_RAT
        elif conv in (v23c.CONVERSION_TYPE_EXPO, v23c.CONVERSION_TYPE_LOGH):
            keys = v23c.KEYS_CONVESION_EXPO_LOGH
        elif conv in (v23c.CONVERSION_TYPE_TABI, v23c.CONVERSION_TYPE_TAB):
            nr = self['ref_param_nr']
            keys = list(v23c.KEYS_CONVESION_NONE)
            for i in range(nr):
                keys.append('raw_{}'.format(i))
                keys.append('phys_{}'.format(i))
        elif conv == v23c.CONVERSION_TYPE_RTABX:
            nr = self['ref_param_nr']
            keys = list(v23c.KEYS_CONVESION_NONE)
            keys += [
                'default_lower',
                'default_upper',
                'default_addr',
            ]
            for i in range(nr - 1):
                keys.append('lower_{}'.format(i))
                keys.append('upper_{}'.format(i))
                keys.append('text_{}'.format(i))
        elif conv == v23c.CONVERSION_TYPE_TABX:
            nr = self['ref_param_nr']
            keys = list(v23c.KEYS_CONVESION_NONE)
            for i in range(nr):
                keys.append('param_val_{}'.format(i))
                keys.append('text_{}'.format(i))

        if self['block_len'] > v23c.MAX_UINT16:
            self['block_len'] = v23c.MAX_UINT16
        result = pack(fmt, *[self[key] for key in keys])
        return result

    def __str__(self):
        return 'ChannelConversion (referneced blocks: {}, address: {}, fields: {})'.format(
            self.referenced_blocks,
            hex(self.address),
            super(ChannelConversion, self).__str__(),
        )


class ChannelDependency(dict):
    ''' CDBLOCK class

    *ChannelDependency* has the following key-value pairs

    * ``id`` - bytes : block ID; always b'CD'
    * ``block_len`` - int : block bytes size
    * ``dependency_type`` - int : integer code for dependency type
    * ``sd_nr`` - int : total number of signals dependencies
    * ``dg_<N>`` - address of data group block (DGBLOCK) of N-th
      signal dependency
    * ``dg_<N>`` - address of channel group block (CGBLOCK) of N-th
      signal dependency
    * ``dg_<N>`` - address of channel block (CNBLOCK) of N-th
      signal dependency
    * ``dim_<K>`` - int : Optional size of dimension *K* for N-dimensional
      dependency

    Parameters
    ----------
    stream : file handle
        mdf file handle
    address : int
        block address inside mdf file
    for dynamically created objects :
        see the key-value pairs

    Attributes
    ----------
    address : int
        block address inside mdf file
    referenced_channels : list
        list of (group index, channel index) pairs

    '''

    def __init__(self, **kwargs):
        super(ChannelDependency, self).__init__()

        self.referenced_channels = []

        try:
            stream = kwargs['stream']
            self.address = address = kwargs['address']
            stream.seek(address)

            (self['id'],
             self['block_len'],
             self['dependency_type'],
             self['sd_nr']) = unpack('<2s3H', stream.read(8))

            links_size = 3 * 4 * self['sd_nr']
            links = unpack(
                '<{}I'.format(3 * self['sd_nr']),
                stream.read(links_size),
            )

            for i in range(self['sd_nr']):
                self['dg_{}'.format(i)] = links[3 * i]
                self['cg_{}'.format(i)] = links[3 * i + 1]
                self['ch_{}'.format(i)] = links[3 * i + 2]

            optional_dims_nr = (self['block_len'] - 8 - links_size) // 2
            if optional_dims_nr:
                dims = unpack(
                    '<{}H'.format(optional_dims_nr),
                    stream.read(optional_dims_nr * 2),
                )
                for i, dim in enumerate(dims):
                    self['dim_{}'.format(i)] = dim

            if self['id'] != b'CD':
                message = 'Expected "CD" block @{} but found "{}"'
                message = message.format(hex(address), self['id'])
                logger.exception(message)
                raise MdfException(message)

        except KeyError:
            sd_nr = kwargs['sd_nr']
            self['id'] = b'CD'
            self['block_len'] = 8 + 3 * 4 * sd_nr
            self['dependency_type'] = 1
            self['sd_nr'] = sd_nr
            for i in range(sd_nr):
                self['dg_{}'.format(i)] = 0
                self['cg_{}'.format(i)] = 0
                self['ch_{}'.format(i)] = 0
            i = 0
            while True:
                try:
                    self['dim_{}'.format(i)] = kwargs['dim_{}'.format(i)]
                    i += 1
                except KeyError:
                    break
            if i:
                self['dependency_type'] = 256 + i
                self['block_len'] += 2 * i

    def __bytes__(self):
        fmt = '<2s3H{}I'.format(self['sd_nr'] * 3)
        keys = ('id', 'block_len', 'dependency_type', 'sd_nr')
        for i in range(self['sd_nr']):
            keys += ('dg_{}'.format(i), 'cg_{}'.format(i), 'ch_{}'.format(i))
        links_size = 3 * 4 * self['sd_nr']
        option_dims_nr = (self['block_len'] - 8 - links_size) // 2
        if option_dims_nr:
            fmt += '{}H'.format(option_dims_nr)
            keys += tuple('dim_{}'.format(i) for i in range(option_dims_nr))
        result = pack(fmt, *[self[key] for key in keys])
        return result


class ChannelExtension(dict):
    ''' CEBLOCK class

    *Channel* has the following common key-value pairs

    * ``id`` - bytes : block ID; always b'CE'
    * ``block_len`` - int : block bytes size
    * ``type`` - int : extension type identifier

    *Channel* has the following specific key-value pairs

    * for DIM block

        * ``module_nr`` - int: module number
        * ``module_address`` - int : module address
        * ``description`` - bytes : module description
        * ``ECU_identification`` - bytes : identification of ECU
        * ``reserved0`` - bytes : reserved bytes

    * for Vector CAN block

        * ``CAN_id`` - int : CAN message ID
        * ``CAN_ch_index`` - int : index of CAN channel
        * ``message_name`` - bytes : message name
        * ``sender_name`` - btyes : sender name
        * ``reserved0`` - bytes : reserved bytes

    Parameters
    ----------
    stream : file handle
        mdf file handle
    address : int
        block address inside mdf file
    for dynamically created objects :
        see the key-value pairs

    Attributes
    ----------
    address : int
        block address inside mdf file
    comment : str
        extension comment
    name : str
        extension name
    path : str
        extension path

    '''

    def __init__(self, **kwargs):
        super(ChannelExtension, self).__init__()

        self.name = self.path = self.comment = ''

        if 'stream' in kwargs:
            stream = kwargs['stream']
            try:

                (self['id'],
                 self['block_len'],
                 self['type']) = unpack_from(
                    v23c.FMT_SOURCE_COMMON,
                    kwargs['raw_bytes'],
                )
                if self['type'] == v23c.SOURCE_ECU:
                    (self['module_nr'],
                     self['module_address'],
                     self['description'],
                     self['ECU_identification'],
                     self['reserved0']) = unpack_from(
                        v23c.FMT_SOURCE_EXTRA_ECU,
                        kwargs['raw_bytes'],
                        6,
                    )
                elif self['type'] == v23c.SOURCE_VECTOR:
                    (self['CAN_id'],
                     self['CAN_ch_index'],
                     self['message_name'],
                     self['sender_name'],
                     self['reserved0']) = unpack_from(
                        v23c.FMT_SOURCE_EXTRA_VECTOR,
                        kwargs['raw_bytes'],
                        6,
                    )

                self.address = kwargs.get('address', 0)
            except KeyError:

                self.address = address = kwargs['address']
                stream.seek(address)
                (self['id'],
                 self['block_len'],
                 self['type']) = unpack(v23c.FMT_SOURCE_COMMON, stream.read(6))
                block = stream.read(self['block_len'] - 6)

                if self['type'] == v23c.SOURCE_ECU:
                    (self['module_nr'],
                     self['module_address'],
                     self['description'],
                     self['ECU_identification'],
                     self['reserved0']) = unpack(
                        v23c.FMT_SOURCE_EXTRA_ECU,
                        block,
                    )
                elif self['type'] == v23c.SOURCE_VECTOR:
                    (self['CAN_id'],
                     self['CAN_ch_index'],
                     self['message_name'],
                     self['sender_name'],
                     self['reserved0']) = unpack(
                        v23c.FMT_SOURCE_EXTRA_VECTOR,
                        block,
                    )

            if self['id'] != b'CE':
                message = 'Expected "CE" block @{} but found "{}"'
                message = message.format(hex(address), self['id'])
                logger.exception(message)
                raise MdfException(message)

        else:

            self.address = 0
            self['id'] = b'CE'
            self['block_len'] = kwargs.get('block_len', v23c.CE_BLOCK_SIZE)
            self['type'] = kwargs.get('type', 2)
            if self['type'] == v23c.SOURCE_ECU:
                self['module_nr'] = kwargs.get('module_nr', 0)
                self['module_address'] = kwargs.get('module_address', 0)
                self['description'] = kwargs.get('description', b'\0')
                self['ECU_identification'] = kwargs.get(
                    'ECU_identification',
                    b'\0',
                )
                self['reserved0'] = kwargs.get('reserved0', b'\0')
            elif self['type'] == v23c.SOURCE_VECTOR:
                self['CAN_id'] = kwargs.get('CAN_id', 0)
                self['CAN_ch_index'] = kwargs.get('CAN_ch_index', 0)
                self['message_name'] = kwargs.get('message_name', b'\0')
                self['sender_name'] = kwargs.get('sender_name', b'\0')
                self['reserved0'] = kwargs.get('reserved0', b'\0')

        if self['type'] == v23c.SOURCE_ECU:
            self.path = self['ECU_identification'].decode('latin-1').strip(' \t\n\r\0')
            self.name = self['description'].decode('latin-1').strip(' \t\n\r\0')
            self.comment = 'Module number={} @ address={}'.format(
                self['module_nr'],
                self['module_address'],
            )
        else:
            self.path = self['sender_name'].decode('latin-1').strip(' \t\n\r\0')
            self.name = self['message_name'].decode('latin-1').strip(' \t\n\r\0')
            self.comment = 'Message ID={} on CAN bus {}'.format(
                hex(self['CAN_id']),
                self['CAN_ch_index'],
            )

    def to_blocks(self, address, blocks, defined_texts, cc_map):

        if self['type'] == v23c.SOURCE_ECU:
            self['ECU_identification'] = self.path.encode('latin-1')[:31]
            self['description'] = self.name.encode('latin-1')[:79]
        else:
            self['sender_name'] = self.path.encode('latin-1')[:35]
            self['message_name'] = self.name.encode('latin-1')[:35]

        bts = bytes(self)
        if bts in cc_map:
            self.address = cc_map[bts]
        else:
            blocks.append(bts)
            self.address = address
            cc_map[bts] = address
            address += self['block_len']

        return address

    def to_stream(self, stream, defined_texts, cc_map):
        address = stream.tell()

        if self['type'] == v23c.SOURCE_ECU:
            self['ECU_identification'] = self.path.encode('latin-1')[:31]
            self['description'] = self.name.encode('latin-1')[:79]
        else:
            self['sender_name'] = self.path.encode('latin-1')[:35]
            self['message_name'] = self.name.encode('latin-1')[:35]

        bts = bytes(self)
        if bts in cc_map:
            self.address = cc_map[bts]
        else:
            cc_map[bts] = address
            stream.write(bytes(self))
            self.address = address
            address += self['block_len']

        return address

    def to_common_source(self):
        if self['type'] == v23c.SOURCE_ECU:
            source = SignalSource(
                self.name,
                self.path,
                self.comment,
                0,  # source type other
                0,  # bus type none
            )
        else:
            source = SignalSource(
                self.name,
                self.path,
                self.comment,
                2,  # source type bus
                2,  # bus type
            )
        return source

    def metadata(self):
        max_len = max(
            len(key)
            for key in self
        )
        template = '{{: <{}}}: {{}}'.format(max_len)

        metadata = []
        lines = """
address: {}

""".format(
            hex(self.address),
        ).split('\n')
        for key, val in self.items():
            if key.endswith('addr') or key.startswith('text_'):
                lines.append(
                    template.format(key, hex(val))
                )
            elif isinstance(val, float):
                    lines.append(
                        template.format(key, round(val, 6))
                    )
            else:
                if (PYVERSION < 3 and isinstance(val, str)) or \
                        (PYVERSION >= 3 and isinstance(val, bytes)):
                    lines.append(
                        template.format(key, val.strip(b'\0'))
                    )
                else:
                    lines.append(
                        template.format(key, val)
                    )
        for line in lines:
            if not line:
                metadata.append(line)
            else:
                for wrapped_line in wrap(line, width=120):
                    metadata.append(wrapped_line)

        return '\n'.join(metadata)

    def __bytes__(self):
        typ = self['type']
        if typ == v23c.SOURCE_ECU:
            fmt = v23c.FMT_SOURCE_ECU
            keys = v23c.KEYS_SOURCE_ECU
        else:
            fmt = v23c.FMT_SOURCE_VECTOR
            keys = v23c.KEYS_SOURCE_VECTOR

        result = pack(fmt, *[self[key] for key in keys])
        return result

    def __str__(self):
        return 'ChannelExtension (name: {}, path: {}, comment: {}, address: {}, fields: {})'.format(
            self.name,
            self.path,
            self.comment,
            hex(self.address),
            super(ChannelExtension, self).__str__(),
        )


class ChannelGroup(dict):
    ''' CGBLOCK class

    *ChannelGroup* has the following key-value pairs

    * ``id`` - bytes : block ID; always b'CG'
    * ``block_len`` - int : block bytes size
    * ``next_cg_addr`` - int : next CGBLOCK address
    * ``first_ch_addr`` - int : address of first channel block (CNBLOCK)
    * ``comment_addr`` - int : address of TXBLOCK that contains the channel
      group comment
    * ``record_id`` - int : record ID used as identifier for a record if
      the DGBLOCK defines a number of record IDs > 0 (unsorted group)
    * ``ch_nr`` - int : number of channels
    * ``samples_byte_nr`` - int : size of data record in bytes without
      record ID
    * ``cycles_nr`` - int : number of cycles (records) of this type in the data
      block
    * ``sample_reduction_addr`` - int : addresss to first sample reduction
      block

    Parameters
    ----------
    stream : file handle
        mdf file handle
    address : int
        block address inside mdf file
    for dynamically created objects :
        see the key-value pairs

    Attributes
    ----------
    address : int
        block address inside mdf file
    comment : str
        channel group comment

    Examples
    --------
    >>> with open('test.mdf', 'rb') as mdf:
    ...     cg1 = ChannelGroup(stream=mdf, address=0xBA52)
    >>> cg2 = ChannelGroup(sample_bytes_nr=32)
    >>> hex(cg1.address)
    0xBA52
    >>> cg1['id']
    b'CG'

    '''

    def __init__(self, **kwargs):
        super(ChannelGroup, self).__init__()
        self.comment = ''

        try:

            stream = kwargs['stream']
            self.address = address = kwargs['address']
            stream.seek(address)
            block = stream.read(v23c.CG_PRE_330_BLOCK_SIZE)

            (self['id'],
             self['block_len'],
             self['next_cg_addr'],
             self['first_ch_addr'],
             self['comment_addr'],
             self['record_id'],
             self['ch_nr'],
             self['samples_byte_nr'],
             self['cycles_nr']) = unpack(v23c.FMT_CHANNEL_GROUP, block)
            if self['block_len'] == v23c.CG_POST_330_BLOCK_SIZE:
                self['sample_reduction_addr'] = unpack('<I', stream.read(4))[0]
                # sample reduction blocks are not yet used
                self['sample_reduction_addr'] = 0
            if self['id'] != b'CG':
                message = 'Expected "CG" block @{} but found "{}"'
                message = message.format(hex(address), self['id'])
                raise MdfException(message.format(self['id']))
            if self['comment_addr']:
                self.comment = get_text_v3(
                    address=self['comment_addr'],
                    stream=stream,
                )
        except KeyError:
            self.address = 0
            self['id'] = b'CG'
            self['block_len'] = kwargs.get(
                'block_len',
                v23c.CG_PRE_330_BLOCK_SIZE,
            )
            self['next_cg_addr'] = kwargs.get('next_cg_addr', 0)
            self['first_ch_addr'] = kwargs.get('first_ch_addr', 0)
            self['comment_addr'] = kwargs.get('comment_addr', 0)
            self['record_id'] = kwargs.get('record_id', 1)
            self['ch_nr'] = kwargs.get('ch_nr', 0)
            self['samples_byte_nr'] = kwargs.get('samples_byte_nr', 0)
            self['cycles_nr'] = kwargs.get('cycles_nr', 0)
            if self['block_len'] == v23c.CG_POST_330_BLOCK_SIZE:
                self['sample_reduction_addr'] = 0

    def to_blocks(self, address, blocks, defined_texts, si_map):
        key = 'comment_addr'
        text = self.comment
        if text:
            if text in defined_texts:
                self[key] = defined_texts[text]
            else:
                tx_block = TextBlock(text=text)
                self[key] = address
                defined_texts[text] = address
                tx_block.address = address
                address += tx_block['block_len']
                blocks.append(tx_block)
        else:
            self[key] = 0

        blocks.append(self)
        self.address = address
        address += self['block_len']

        return address

    def to_stream(self, stream, defined_texts, si_map):
        address = stream.tell()

        key = 'comment_addr'
        text = self.comment
        if text:
            if text in defined_texts:
                self[key] = defined_texts[text]
            else:
                tx_block = TextBlock(text=text)
                self[key] = address
                defined_texts[text] = address
                tx_block.address = address
                address += tx_block['block_len']
                stream.write(bytes(tx_block))
        else:
            self[key] = 0

        stream.write(bytes(self))
        self.address = address
        address += self['block_len']

        return address

    def __bytes__(self):
        fmt = v23c.FMT_CHANNEL_GROUP
        keys = v23c.KEYS_CHANNEL_GROUP
        if self['block_len'] == v23c.CG_POST_330_BLOCK_SIZE:
            fmt += 'I'
            keys += ('sample_reduction_addr',)
        result = pack(fmt, *[self[key] for key in keys])
        return result


class DataBlock(dict):
    """Data Block class (pseudo block not defined by the MDF 3 standard)

    *DataBlock* has the following key-value pairs

    * ``data`` - bytes : raw samples bytes

    Attributes
    ----------
    address : int
        block address

    Parameters
    ----------
    address : int
        block address inside the measurement file
    stream : file.io.handle
        binary file stream
    data : bytes
        when created dynamically

    """

    def __init__(self, **kwargs):
        super(DataBlock, self).__init__()

        try:
            stream = kwargs['stream']
            size = kwargs['size']
            self.address = address = kwargs['address']
            stream.seek(address)

            self['data'] = stream.read(size)

        except KeyError:
            self.address = 0
            self['data'] = kwargs.get('data', b'')

    def __bytes__(self):
        return self['data']


class DataGroup(dict):
    ''' DGBLOCK class

    *Channel* has the following key-value pairs

    * ``id`` - bytes : block ID; always b'DG'
    * ``block_len`` - int : block bytes size
    * ``next_dg_addr`` - int : next DGBLOCK address
    * ``first_cg_addr`` - int : address of first channel group block (CGBLOCK)
    * ``trigger_addr`` - int : address of trigger block (TRBLOCK)
    * ``data_block_addr`` - addrfss of data block
    * ``cg_nr`` - int : number of channel groups
    * ``record_id_len`` - int : number of record IDs in the data block
    * ``reserved0`` - bytes : reserved bytes

    Parameters
    ----------
    stream : file handle
        mdf file handle
    address : int
        block address inside mdf file
    for dynamically created objects :
        see the key-value pairs

    Attributes
    ----------
    address : int
        block address inside mdf file

    '''

    def __init__(self, **kwargs):
        super(DataGroup, self).__init__()

        try:
            stream = kwargs['stream']
            self.address = address = kwargs['address']
            stream.seek(address)
            block = stream.read(v23c.DG_PRE_320_BLOCK_SIZE)

            (self['id'],
             self['block_len'],
             self['next_dg_addr'],
             self['first_cg_addr'],
             self['trigger_addr'],
             self['data_block_addr'],
             self['cg_nr'],
             self['record_id_len']) = unpack(v23c.FMT_DATA_GROUP_PRE_320, block)

            if self['block_len'] == v23c.DG_POST_320_BLOCK_SIZE:
                self['reserved0'] = stream.read(4)

            if self['id'] != b'DG':
                message = 'Expected "DG" block @{} but found "{}"'
                message = message.format(hex(address), self['id'])
                logger.exception(message)
                raise MdfException(message)

        except KeyError:
            self.address = 0
            self['id'] = b'DG'
            self['block_len'] = kwargs.get(
                'block_len',
                v23c.DG_PRE_320_BLOCK_SIZE,
            )
            self['next_dg_addr'] = kwargs.get('next_dg_addr', 0)
            self['first_cg_addr'] = kwargs.get('first_cg_addr', 0)
            self['trigger_addr'] = kwargs.get('comment_addr', 0)
            self['data_block_addr'] = kwargs.get('data_block_addr', 0)
            self['cg_nr'] = kwargs.get('cg_nr', 1)
            self['record_id_len'] = kwargs.get('record_id_len', 0)
            if self['block_len'] == v23c.DG_POST_320_BLOCK_SIZE:
                self['reserved0'] = b'\0\0\0\0'

    def __bytes__(self):
        if self['block_len'] == v23c.DG_POST_320_BLOCK_SIZE:
            fmt = v23c.FMT_DATA_GROUP_POST_320
            keys = v23c.KEYS_DATA_GROUP_POST_320
        else:
            fmt = v23c.FMT_DATA_GROUP_PRE_320
            keys = v23c.KEYS_DATA_GROUP_PRE_320
        result = pack(fmt, *[self[key] for key in keys])
        return result


class FileIdentificationBlock(dict):
    ''' IDBLOCK class

    *FileIdentificationBlock* has the following key-value pairs

    * ``file_identification`` -  bytes : file identifier
    * ``version_str`` - bytes : format identifier
    * ``program_identification`` - bytes : creator program identifier
    * ``byte_order`` - int : integer code for byte order (endiannes)
    * ``float_format`` - int : integer code for floating-point format
    * ``mdf_version`` - int : version number of MDF format
    * ``code_page`` - int : unicode code page number
    * ``reserved0`` - bytes : reserved bytes
    * ``reserved1`` - bytes : reserved bytes
    * ``unfinalized_standard_flags`` - int : standard flags for unfinalized MDF
    * ``unfinalized_custom_flags`` - int : custom flags for unfinalized MDF

    Parameters
    ----------
    stream : file handle
        mdf file handle
    version : int
        mdf version in case of new file (dynamically created)

    Attributes
    ----------
    address : int
        block address inside mdf file; should be 0 always

    '''

    def __init__(self, **kwargs):
        super(FileIdentificationBlock, self).__init__()

        self.address = 0
        try:

            stream = kwargs['stream']
            stream.seek(0)

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
             self['unfinalized_custom_flags']) = unpack(
                v23c.ID_FMT,
                stream.read(v23c.ID_BLOCK_SIZE),
            )
        except KeyError:
            version = kwargs['version']
            self['file_identification'] = 'MDF     '.encode('latin-1')
            self['version_str'] = version.encode('latin-1') + b'\0' * 4
            self['program_identification'] = (
                'amdf{}'
                .format(__version__.replace('.', ''))
                .encode('latin-1')
            )
            self['byte_order'] = v23c.BYTE_ORDER_INTEL
            self['float_format'] = 0
            self['mdf_version'] = int(version.replace('.', ''))
            self['code_page'] = 0
            self['reserved0'] = b'\0' * 2
            self['reserved1'] = b'\0' * 26
            self['unfinalized_standard_flags'] = 0
            self['unfinalized_custom_flags'] = 0

    def __bytes__(self):
        result = pack(v23c.ID_FMT, *[self[key] for key in v23c.ID_KEYS])
        return result


class HeaderBlock(dict):
    ''' HDBLOCK class

    *HeaderBlock* has the following key-value pairs

    * ``id`` - bytes : block ID; always b'HD'
    * ``block_len`` - int : block bytes size
    * ``first_dg_addr`` - int : address of first data group block (DGBLOCK)
    * ``comment_addr`` - int : address of TXBLOCK taht contains the
      measurement file comment
    * ``program_addr`` - int : address of program block (PRBLOCK)
    * ``dg_nr`` - int : number of data groups
    * ``date`` - bytes : date at which the recording was started in
      "DD:MM:YYYY" format
    * ``time`` - btyes : time at which the recording was started in
      "HH:MM:SS" format
    * ``author`` - btyes : author name
    * ``organization`` - bytes : organization name
    * ``project`` - bytes : project name
    * ``subject`` - bytes : subject

    Since version 3.2 the following extra keys were added:

    * ``abs_time`` - int : time stamp at which recording was started in
      nanoseconds.
    * ``tz_offset`` - int : UTC time offset in hours (= GMT time zone)
    * ``time_quality`` - int : time quality class
    * ``timer_identification`` - bytes : timer identification (time source)

    Parameters
    ----------
    stream : file handle
        mdf file handle
    version : int
        mdf version in case of new file (dynamically created)

    Attributes
    ----------
    address : int
        block address inside mdf file; should be 64 always
    comment : int
        file comment
    program : ProgramBlock
        program block
    author : str
        measurement author
    department : str
        author's department
    project : str
        working project
    subject : str
        measurement subject

    '''

    def __init__(self, **kwargs):
        super(HeaderBlock, self).__init__()

        self.address = 64
        self.program = None
        self.comment = ''
        try:

            stream = kwargs['stream']
            stream.seek(64)

            (self['id'],
             self['block_len'],
             self['first_dg_addr'],
             self['comment_addr'],
             self['program_addr'],
             self['dg_nr'],
             self['date'],
             self['time'],
             self['author'],
             self['department'],
             self['project'],
             self['subject']) = unpack(
                v23c.HEADER_COMMON_FMT,
                stream.read(v23c.HEADER_COMMON_SIZE),
            )

            if self['block_len'] > v23c.HEADER_COMMON_SIZE:
                (self['abs_time'],
                 self['tz_offset'],
                 self['time_quality'],
                 self['timer_identification']) = unpack(
                    v23c.HEADER_POST_320_EXTRA_FMT,
                    stream.read(v23c.HEADER_POST_320_EXTRA_SIZE),
                )

            if self['id'] != b'HD':
                message = 'Expected "HD" block @{} but found "{}"'
                message = message.format(hex(address), self['id'])
                logger.exception(message)
                raise MdfException(message)

            if self['program_addr']:
                self.program = ProgramBlock(
                    address=self['program_addr'],
                    stream=stream,
                )
            if self['comment_addr']:
                self.comment = get_text_v3(
                    address=self['comment_addr'],
                    stream=stream,
                )

        except KeyError:
            version = kwargs.get('version', '3.20')
            self['id'] = b'HD'
            self['block_len'] = 208 if version >= '3.20' else 164
            self['first_dg_addr'] = 0
            self['comment_addr'] = 0
            self['program_addr'] = 0
            self['dg_nr'] = 0
            t1 = time.time() * 10 ** 9
            t2 = time.gmtime()
            self['date'] = (
                '{:\0<10}'
                .format(time.strftime('%d:%m:%Y', t2))
                .encode('latin-1')
            )
            self['time'] = (
                '{:\0<8}'
                .format(time.strftime('%X', t2))
                .encode('latin-1')
            )
            self['author'] = (
                '{:\0<32}'
                .format(getuser())
                .encode('latin-1')
            )
            self['department'] = (
                '{:\0<32}'
                .format('')
                .encode('latin-1')
            )
            self['project'] = (
                '{:\0<32}'
                .format('')
                .encode('latin-1')
            )
            self['subject'] = (
                '{:\0<32}'
                .format('')
                .encode('latin-1')
            )

            if self['block_len'] > v23c.HEADER_COMMON_SIZE:
                self['abs_time'] = int(t1)
                self['tz_offset'] = 2
                self['time_quality'] = 0
                self['timer_identification'] = (
                    '{:\0<32}'
                    .format('Local PC Reference Time')
                    .encode('latin-1')
                )

        self.author = self['author'].strip(b' \r\n\t\0').decode('latin-1')
        self.department = self['department'].strip(b' \r\n\t\0').decode('latin-1')
        self.project = self['project'].strip(b' \r\n\t\0').decode('latin-1')
        self.subject = self['subject'].strip(b' \r\n\t\0').decode('latin-1')

    def to_blocks(self, address, blocks, defined_texts, si_map):
        blocks.append(self)
        self.address = address
        address += self['block_len']

        key = 'comment_addr'
        text = self.comment
        if text:
            if text in defined_texts:
                self[key] = defined_texts[text]
            else:
                tx_block = TextBlock(text=text)
                self[key] = address
                defined_texts[text] = address
                tx_block.address = address
                address += tx_block['block_len']
                blocks.append(tx_block)
        else:
            self[key] = 0

        key = 'program_addr'
        if self.program:
            self[key] = address
            address += self.program['block_len']
            blocks.append(self.program)

        else:
            self[key] = 0

        self['author'] = self.author.encode('latin-1')
        self['department'] = self.department.encode('latin-1')
        self['project'] = self.project.encode('latin-1')
        self['subject'] = self.subject.encode('latin-1')

        return address

    def to_stream(self, stream, defined_texts, si_map):
        address = start = stream.tell()
        stream.write(bytes(self))
        address += self['block_len']

        key = 'comment_addr'
        text = self.comment
        if text:
            if text in defined_texts:
                self[key] = defined_texts[text]
            else:
                tx_block = TextBlock(text=text)
                self[key] = address
                defined_texts[text] = address
                tx_block.address = address
                address += tx_block['block_len']
                stream.write(bytes(tx_block))
        else:
            self[key] = 0

        key = 'program_addr'
        if self.program:
            self[key] = address
            address += self.program['block_len']
            stream.write(bytes(self.program))

        else:
            self[key] = 0

        self['author'] = self.author.encode('latin-1')
        self['department'] = self.department.encode('latin-1')
        self['project'] = self.project.encode('latin-1')
        self['subject'] = self.subject.encode('latin-1')

        stream.seek(start)
        stream.write(bytes(self))
        self.address = address
        address += self['block_len']

        return address

    @property
    def start_time(self):
        """ getter and setter the measurement start timestamp

        Returns
        -------
        timestamp : datetime.datetime
            start timestamp

        """

        if self['block_len'] > v23c.HEADER_COMMON_SIZE:
            timestamp = self['abs_time'] / 10 ** 9
            try:
                timestamp = datetime.fromtimestamp(timestamp)
            except OSError:
                timestamp = datetime.now()

        else:
            timestamp = '{} {}'.format(
                self['date'].decode('ascii'),
                self['time'].decode('ascii'),
            )

            timestamp = datetime.strptime(
                timestamp,
                '%d:%m:%Y %H:%M:%S',
            )

        return timestamp

    @start_time.setter
    def start_time(self, timestamp):
        self['date'] = timestamp.strftime('%d:%m:%Y').encode('ascii')
        self['time'] = timestamp.strftime('%H:%M:%S').encode('ascii')
        if self['block_len'] > v23c.HEADER_COMMON_SIZE:
            timestamp = timestamp - datetime(1970, 1, 1)
            timestamp = int(timestamp.total_seconds() * 10**9)
            self['abs_time'] = timestamp
            self['tz_offset'] = 0

    def __bytes__(self):
        fmt = v23c.HEADER_COMMON_FMT
        keys = v23c.HEADER_COMMON_KEYS
        if self['block_len'] > v23c.HEADER_COMMON_SIZE:
            fmt += v23c.HEADER_POST_320_EXTRA_FMT
            keys += v23c.HEADER_POST_320_EXTRA_KEYS
        result = pack(fmt, *[self[key] for key in keys])
        return result


class ProgramBlock(dict):
    ''' PRBLOCK class

    *ProgramBlock* has the following key-value pairs

    * ``id`` - bytes : block ID; always b'PR'
    * ``block_len`` - int : block bytes size
    * ``data`` - btyes : creator program free format data

    Parameters
    ----------
    stream : file handle
        mdf file handle
    address : int
        block address inside mdf file

    Attributes
    ----------
    address : int
        block address inside mdf file

    '''

    def __init__(self, **kwargs):
        super(ProgramBlock, self).__init__()

        try:
            stream = kwargs['stream']
            self.address = address = kwargs['address']
            stream.seek(address)

            (self['id'],
             self['block_len']) = unpack('<2sH', stream.read(4))
            self['data'] = stream.read(self['block_len'] - 4)

            if self['id'] != b'PR':
                message = 'Expected "PR" block @{} but found "{}"'
                message = message.format(hex(address), self['id'])
                logger.exception(message)
                raise MdfException(message)

        except KeyError:
            self['id'] = b'PR'
            self['block_len'] = len(kwargs['data']) + 6
            self['data'] = kwargs['data']

    def __bytes__(self):
        fmt = v23c.FMT_PROGRAM_BLOCK.format(self['block_len'])
        result = pack(fmt, *[self[key] for key in v23c.KEYS_PROGRAM_BLOCK])
        return result


class SampleReduction(dict):
    ''' SRBLOCK class

    *SampleReduction* has the following key-value pairs

    * ``id`` - bytes : block ID; always b'SR'
    * ``block_len`` - int : block bytes size
    * ``next_sr_addr`` - int : next SRBLOCK address
    * ``data_block_addr`` - int : address of data block for this sample
      reduction
    * ``cycles_nr`` - int : number of reduced samples in the data block
    * ``time_interval`` - float : length of time interval [s] used to calculate
        the reduced samples

    Parameters
    ----------
    stream : file handle
        mdf file handle
    address : int
        block address inside mdf file

    Attributes
    ----------
    address : int
        block address inside mdf file

    '''

    def __init__(self, **kwargs):
        super(SampleReduction, self).__init__()

        try:
            stream = kwargs['stream']
            self.address = address = kwargs['address']
            stream.seek(address)

            (self['id'],
             self['block_len'],
             self['next_sr_addr'],
             self['data_block_addr'],
             self['cycles_nr'],
             self['time_interval']) = unpack(
                v23c.FMT_SAMPLE_REDUCTION_BLOCK,
                stream.read(v23c.SR_BLOCK_SIZE),
            )

            if self['id'] != b'SR':
                message = 'Expected "SR" block @{} but found "{}"'
                message = message.format(hex(address), self['id'])
                logger.exception(message)
                raise MdfException(message)

        except KeyError:
            pass

    def __bytes__(self):
        result = pack(
            v23c.FMT_SAMPLE_REDUCTION_BLOCK,
            *[self[key] for key in v23c.KEYS_SAMPLE_REDUCTION_BLOCK]
        )
        return result


class TextBlock(dict):
    ''' TXBLOCK class

    *TextBlock* has the following key-value pairs

    * ``id`` - bytes : block ID; always b'TX'
    * ``block_len`` - int : block bytes size
    * ``text`` - bytes : text content

    Parameters
    ----------
    stream : file handle
        mdf file handle
    address : int
        block address inside mdf file
    text : bytes | str
        bytes or str for creating a new TextBlock

    Attributes
    ----------
    address : int
        block address inside mdf file

    Examples
    --------
    >>> tx1 = TextBlock(text='VehicleSpeed')
    >>> tx1.text_str
    'VehicleSpeed'
    >>> tx1['text']
    b'VehicleSpeed'

    '''

    def __init__(self, **kwargs):
        super(TextBlock, self).__init__()
        try:

            stream = kwargs['stream']
            self.address = address = kwargs['address']
            stream.seek(address)
            (self['id'],
             self['block_len']) = unpack('<2sH', stream.read(4))
            size = self['block_len'] - 4
            self['text'] = stream.read(size)

            if self['id'] != b'TX':
                message = 'Expected "TX" block @{} but found "{}"'
                message = message.format(hex(address), self['id'])
                logger.exception(message)
                raise MdfException(message)

        except KeyError:
            self.address = 0
            text = kwargs['text']

            if PYVERSION == 3:
                try:
                    text = text.encode('utf-8')
                except AttributeError:
                    pass
            else:
                try:
                    text = text.encode('utf-8')
                except (AttributeError, UnicodeDecodeError):
                    pass

            self['id'] = b'TX'
            self['block_len'] = len(text) + 4 + 1
            self['text'] = text + b'\0'

    def __bytes__(self):
        result = pack(
            '<2sH{}s'.format(self['block_len'] - 4),
            *[self[key] for key in v23c.KEYS_TEXT_BLOCK]
        )
        return result


class TriggerBlock(dict):
    ''' TRBLOCK class

    *Channel* has the following key-value pairs

    * ``id`` - bytes : block ID; always b'TR'
    * ``block_len`` - int : block bytes size
    * ``text_addr`` - int : address of TXBLOCK that contains the trigger
      comment text
    * ``trigger_events_nr`` - int : number of trigger events
    * ``trigger_<N>_time`` - float : trigger time [s] of trigger's N-th event
    * ``trigger_<N>_pretime`` - float : pre trigger time [s] of trigger's N-th
      event
    * ``trigger_<N>_posttime`` - float : post trigger time [s] of trigger's
      N-th event

    Parameters
    ----------
    stream : file handle
        mdf file handle
    address : int
        block address inside mdf file

    Attributes
    ----------
    address : int
        block address inside mdf file
    comment : str
        trigger comment

    '''

    def __init__(self, **kwargs):
        super(TriggerBlock, self).__init__()

        self.comment = ''

        try:
            self.address = address = kwargs['address']
            stream = kwargs['stream']

            stream.seek(address + 2)
            size = unpack('<H', stream.read(2))[0]
            stream.seek(address)
            block = stream.read(size)

            (self['id'],
             self['block_len'],
             self['text_addr'],
             self['trigger_events_nr']) = unpack('<2sHIH', block[:10])

            nr = self['trigger_events_nr']
            if nr:
                values = unpack('<{}d'.format(3 * nr), block[10:])
            for i in range(nr):
                (self['trigger_{}_time'.format(i)],
                 self['trigger_{}_pretime'.format(i)],
                 self['trigger_{}_posttime'.format(i)]) = (
                    values[i * 3],
                    values[3 * i + 1],
                    values[3 * i + 2],
                )

            if self['text_addr']:
                self.comment = get_text_v3(
                    address=self['text_addr'],
                    stream=stream,
                )

            if self['id'] != b'TR':
                message = 'Expected "TR" block @{} but found "{}"'
                message = message.format(hex(address), self['id'])
                logger.exception(message)
                raise MdfException(message)

        except KeyError:
            self.address = 0
            nr = 0
            while 'trigger_{}_time'.format(nr) in kwargs:
                nr += 1

            self['id'] = b'TR'
            self['block_len'] = 10 + nr * 8 * 3
            self['text_addr'] = 0
            self['trigger_events_nr'] = nr

            for i in range(nr):
                key = 'trigger_{}_time'.format(i)
                self[key] = kwargs[key]
                key = 'trigger_{}_pretime'.format(i)
                self[key] = kwargs[key]
                key = 'trigger_{}_posttime'.format(i)
                self[key] = kwargs[key]

    def to_blocks(self, address, blocks):
        key = 'text_addr'
        text = self.comment
        if text:
            tx_block = TextBlock(text=text)
            self[key] = address
            address += tx_block['block_len']
            blocks.append(tx_block)
        else:
            self[key] = 0

        blocks.append(self)
        self.address = address
        address += self['block_len']

        return address

    def to_stream(self, stream):
        address = stream.tell()

        key = 'text_addr'
        text = self.comment
        if text:
            tx_block = TextBlock(text=text)
            self[key] = address
            address += tx_block['block_len']
            stream.write(bytes(tx_block))
        else:
            self[key] = 0

        stream.write(bytes(self))
        self.address = address
        address += self['block_len']

        return address

    def __bytes__(self):
        triggers_nr = self['trigger_events_nr']
        fmt = '<2sHIH{}d'.format(triggers_nr * 3)
        keys = (
            'id',
            'block_len',
            'text_addr',
            'trigger_events_nr',
        )
        for i in range(triggers_nr):
            keys += (
                'trigger_{}_time'.format(i),
                'trigger_{}_pretime'.format(i),
                'trigger_{}_posttime'.format(i),
            )
        result = pack(fmt, *[self[key] for key in keys])
        return result

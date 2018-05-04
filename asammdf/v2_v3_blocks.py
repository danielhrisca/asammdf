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
from .utils import MdfException, get_text_v3

PYVERSION = sys.version_info[0]
PYVERSION_MAJOR = sys.version_info[0] * 10 + sys.version_info[1]
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
    ''' CNBLOCK class derived from *dict*

    The Channel object can be created in two modes:

    * using the *stream* and *address* keyword parameters - when reading
        from file
    * using any of the following presented keys - when creating a new Channel

    The keys have the following meaning:

    * id - Block type identifier, always "CN"
    * block_len - Block size of this block in bytes (entire CNBLOCK)
    * next_ch_addr - Pointer to next channel block (CNBLOCK) of this channel
        group (NIL allowed)
    * conversion_addr - Pointer to the conversion formula (CCBLOCK) of this
        signal (NIL allowed)
    * source_depend_addr - Pointer to the source-depending extensions (CEBLOCK)
        of this signal (NIL allowed)
    * ch_depend_addr - Pointer to the dependency block (CDBLOCK) of this signal
        (NIL allowed)
    * comment_addr - Pointer to the channel comment (TXBLOCK) of this signal
        (NIL allowed)
    * channel_type - Channel type

        * 0 = data channel
        * 1 = time channel for all signals of this group (in each channel
            group, exactly one channel must be defined as time channel).
            The time stamps recording in a time channel are always relative
            to the start time of the measurement defined in HDBLOCK.

    * short_name - Short signal name, i.e. the first 31 characters of the
        ASAM-MCD name of the signal (end of text should be indicated by 0)
    * description - Signal description (end of text should be indicated by 0)
    * start_offset - Start offset in bits to determine the first bit of the
        signal in the data record. The start offset N is divided into two
        parts: a "Byte offset" (= N div 8) and a "Bit offset" (= N mod 8).
        The channel block can define an "additional Byte offset" (see below)
        which must be added to the Byte offset.
    * bit_count - Number of bits used to encode the value of this signal in a
        data record
    * data_type - Signal data type
    * range_flag - Value range valid flag
    * min_raw_value - Minimum signal value that occurred for this signal
        (raw value)
    * max_raw_value - Maximum signal value that occurred for this signal
        (raw value)
    * sampling_rate - Sampling rate for a virtual time channel. Unit [s]
    * long_name_addr - Pointer to TXBLOCK that contains the ASAM-MCD long
        signal name
    * display_name_addr - Pointer to TXBLOCK that contains the signal's display
        name (NIL allowed)
    * aditional_byte_offset - Additional Byte offset of the signal in the data
        record (default value: 0).

    Parameters
    ----------
    stream : file handle
        mdf file handle
    address : int
        block address inside mdf file

    Attributes
    ----------
    name : str
        full channel name
    address : int
        block address inside mdf file
    dependencies : list
        lsit of channel dependencies

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

    def __init__(self, **kargs):
        super(Channel, self).__init__()

        self.name = self.display_name = self.comment = ''

        try:
            stream = kargs['stream']
            self.address = address = kargs['address']
            stream.seek(address + 2)
            size = unpack('<H', stream.read(2))[0]
            stream.seek(address)
            block = stream.read(size)

            if size == v23c.CN_DISPLAYNAME_BLOCK_SIZE:
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
                 self['aditional_byte_offset']) = unpack(
                    v23c.FMT_CHANNEL_DISPLAYNAME,
                    block,
                )

            elif size == v23c.CN_LONGNAME_BLOCK_SIZE:
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
                 self['long_name_addr']) = unpack(
                    v23c.FMT_CHANNEL_LONGNAME,
                    block,
                )
            else:
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
                 self['sampling_rate']) = unpack(v23c.FMT_CHANNEL_SHORT, block)

            if self['id'] != b'CN':
                message = 'Expected "CN" block but found "{}"'
                message = message.format(self['id'])
                logger.exception(message)
                raise MdfException(message)

        except KeyError:

            self.address = 0
            self['id'] = b'CN'
            self['block_len'] = kargs.get(
                'block_len',
                v23c.CN_DISPLAYNAME_BLOCK_SIZE,
            )
            self['next_ch_addr'] = kargs.get('next_ch_addr', 0)
            self['conversion_addr'] = kargs.get('conversion_addr', 0)
            self['source_depend_addr'] = kargs.get('source_depend_addr', 0)
            self['ch_depend_addr'] = kargs.get('ch_depend_addr', 0)
            self['comment_addr'] = kargs.get('comment_addr', 0)
            self['channel_type'] = kargs.get('channel_type', 0)
            self['short_name'] = kargs.get('short_name', (b'\0' * 32))
            self['description'] = kargs.get('description', (b'\0' * 32))
            self['start_offset'] = kargs.get('start_offset', 0)
            self['bit_count'] = kargs.get('bit_count', 8)
            self['data_type'] = kargs.get('data_type', 0)
            self['range_flag'] = kargs.get('range_flag', 1)
            self['min_raw_value'] = kargs.get('min_raw_value', 0)
            self['max_raw_value'] = kargs.get('max_raw_value', 0)
            self['sampling_rate'] = kargs.get('sampling_rate', 0)
            if self['block_len'] >= v23c.CN_LONGNAME_BLOCK_SIZE:
                self['long_name_addr'] = kargs.get('long_name_addr', 0)
            if self['block_len'] >= v23c.CN_DISPLAYNAME_BLOCK_SIZE:
                self['display_name_addr'] = kargs.get('display_name_addr', 0)
                self['aditional_byte_offset'] = kargs.get(
                    'aditional_byte_offset',
                    0,
                )

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

        if PYVERSION_MAJOR >= 36:
            result = pack(fmt, *self.values())
        else:
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

    def __str__(self):
        return 'Channel (name: {}, display name: {}, comment: {}, address: {}, fields: {})'.format(
            self.name,
            self.display_name,
            self.comment,
            hex(self.address),
            super(Channel, self).__str__(),
        )


class ChannelConversion(dict):
    ''' CCBLOCK class derived from *dict*

    The ChannelConversion object can be created in two modes:

    * using the *stream* and *address* keyword parameters - when reading
        from file
    * using any of the following presented keys - when creating a new
        ChannelConversion

    The first keys are common for all conversion types, and are followed by
    conversion specific keys. The keys have the following meaning:

    * common keys

        * id - Block type identifier, always "CC"
        * block_len - Block size of this block in bytes (entire CCBLOCK)
        * range_flag - Physical value range valid flag:
        * min_phy_value - Minimum physical signal value that occurred for this
            signal
        * max_phy_value - Maximum physical signal value that occurred for this
            signal
        * unit - Physical unit (string should be terminated with 0)
        * conversion_type - Conversion type (formula identifier)
        * ref_param_nr - Size information about additional conversion data

    * specific keys

        * linear conversion

            * b - offset
            * a - factor
            * CANapeHiddenExtra - sometimes CANape appends extra information;
                not compliant with MDF specs

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
    stream : file handle
        mdf file handle
    address : int
        block address inside mdf file

    Attributes
    ----------
    address : int
        block address inside mdf file

    Examples
    --------
    >>> with open('test.mdf', 'rb') as mdf:
    ...     cc1 = ChannelConversion(stream=mdf, address=0xBA52)
    >>> cc2 = ChannelConversion(conversion_type=0)
    >>> cc1['b'], cc1['a']
    0, 100.0

    '''

    def __init__(self, **kargs):
        super(ChannelConversion, self).__init__()

        self.referenced_blocks = {}

        if 'raw_bytes' in kargs or 'stream' in kargs:
            try:
                self.address = 0
                block = kargs['raw_bytes']
                (self['id'],
                 self['block_len']) = unpack_from(
                    '<2sH',
                    block,
                )
                size = self['block_len']
                block_size = len(block)
                block = block[4:]
                stream=kargs['stream']

            except KeyError:
                stream = kargs['stream']
                self.address = address = kargs['address']
                stream.seek(address)
                block = stream.read(4)
                (self['id'],
                 self['block_len']) = unpack('<2sH', block)

                size = self['block_len']
                block_size = size
                block = stream.read(size - 4)

            address = kargs.get('address', 0)
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
                message = 'Expected "CC" block but found "{}"'
                message = message.format(self['id'])
                logger.exception(message)
                raise MdfException(message)

        else:

            self.address = 0
            self['id'] = 'CC'.encode('latin-1')

            if kargs['conversion_type'] == v23c.CONVERSION_TYPE_NONE:
                self['block_len'] = v23c.CC_COMMON_BLOCK_SIZE
                self['range_flag'] = kargs.get('range_flag', 1)
                self['min_phy_value'] = kargs.get('min_phy_value', 0)
                self['max_phy_value'] = kargs.get('max_phy_value', 0)
                self['unit'] = kargs.get('unit', ('\0' * 20).encode('latin-1'))
                self['conversion_type'] = v23c.CONVERSION_TYPE_NONE
                self['ref_param_nr'] = 0

            elif kargs['conversion_type'] == v23c.CONVERSION_TYPE_LINEAR:
                self['block_len'] = v23c.CC_LIN_BLOCK_SIZE
                self['range_flag'] = kargs.get('range_flag', 1)
                self['min_phy_value'] = kargs.get('min_phy_value', 0)
                self['max_phy_value'] = kargs.get('max_phy_value', 0)
                self['unit'] = kargs.get('unit', ('\0' * 20).encode('latin-1'))
                self['conversion_type'] = v23c.CONVERSION_TYPE_LINEAR
                self['ref_param_nr'] = 2
                self['b'] = kargs.get('b', 0)
                self['a'] = kargs.get('a', 1)
                if not self['block_len'] == v23c.CC_LIN_BLOCK_SIZE:
                    self['CANapeHiddenExtra'] = kargs['CANapeHiddenExtra']

            elif kargs['conversion_type'] in (
                    v23c.CONVERSION_TYPE_POLY,
                    v23c.CONVERSION_TYPE_RAT):
                self['block_len'] = v23c.CC_POLY_BLOCK_SIZE
                self['range_flag'] = kargs.get('range_flag', 1)
                self['min_phy_value'] = kargs.get('min_phy_value', 0)
                self['max_phy_value'] = kargs.get('max_phy_value', 0)
                self['unit'] = kargs.get('unit', ('\0' * 20).encode('latin-1'))
                self['conversion_type'] = kargs['conversion_type']
                self['ref_param_nr'] = 6
                self['P1'] = kargs.get('P1', 0)
                self['P2'] = kargs.get('P2', 0)
                self['P3'] = kargs.get('P3', 0)
                self['P4'] = kargs.get('P4', 0)
                self['P5'] = kargs.get('P5', 0)
                self['P6'] = kargs.get('P6', 0)

            elif kargs['conversion_type'] in (
                    v23c.CONVERSION_TYPE_EXPO,
                    v23c.CONVERSION_TYPE_LOGH):
                self['block_len'] = v23c.CC_EXPO_BLOCK_SIZE
                self['range_flag'] = kargs.get('range_flag', 1)
                self['min_phy_value'] = kargs.get('min_phy_value', 0)
                self['max_phy_value'] = kargs.get('max_phy_value', 0)
                self['unit'] = kargs.get('unit', ('\0' * 20).encode('latin-1'))
                self['conversion_type'] = v23c.CONVERSION_TYPE_EXPO
                self['ref_param_nr'] = 7
                self['P1'] = kargs.get('P1', 0)
                self['P2'] = kargs.get('P2', 0)
                self['P3'] = kargs.get('P3', 0)
                self['P4'] = kargs.get('P4', 0)
                self['P5'] = kargs.get('P5', 0)
                self['P6'] = kargs.get('P6', 0)
                self['P7'] = kargs.get('P7', 0)

            elif kargs['conversion_type'] == v23c.CONVERSION_TYPE_FORMULA:
                formula = kargs['formula']
                formula_len = len(formula)
                try:
                    formula += b'\0'
                except:
                    formula = formula.encode('latin-1') + b'\0'
                self['block_len'] = 46 + formula_len + 1
                self['range_flag'] = kargs.get('range_flag', 1)
                self['min_phy_value'] = kargs.get('min_phy_value', 0)
                self['max_phy_value'] = kargs.get('max_phy_value', 0)
                self['unit'] = kargs.get('unit', ('\0' * 20).encode('latin-1'))
                self['conversion_type'] = v23c.CONVERSION_TYPE_FORMULA
                self['ref_param_nr'] = formula_len
                self['formula'] = formula

            elif kargs['conversion_type'] in (
                    v23c.CONVERSION_TYPE_TABI,
                    v23c.CONVERSION_TYPE_TAB):
                nr = kargs['ref_param_nr']
                self['block_len'] = v23c.CC_COMMON_BLOCK_SIZE + nr * 2 * 8
                self['range_flag'] = kargs.get('range_flag', 1)
                self['min_phy_value'] = kargs.get('min_phy_value', 0)
                self['max_phy_value'] = kargs.get('max_phy_value', 0)
                self['unit'] = kargs.get('unit', ('\0' * 20).encode('latin-1'))
                self['conversion_type'] = kargs['conversion_type']
                self['ref_param_nr'] = nr
                for i in range(nr):
                    self['raw_{}'.format(i)] = kargs['raw_{}'.format(i)]
                    self['phys_{}'.format(i)] = kargs['phys_{}'.format(i)]

            elif kargs['conversion_type'] == v23c.CONVERSION_TYPE_TABX:
                nr = kargs['ref_param_nr']
                self['block_len'] = v23c.CC_COMMON_BLOCK_SIZE + 40 * nr
                self['range_flag'] = kargs.get('range_flag', 0)
                self['min_phy_value'] = kargs.get('min_phy_value', 0)
                self['max_phy_value'] = kargs.get('max_phy_value', 0)
                self['unit'] = kargs.get('unit', ('\0' * 20).encode('latin-1'))
                self['conversion_type'] = v23c.CONVERSION_TYPE_TABX
                self['ref_param_nr'] = nr

                for i in range(nr):
                    self['param_val_{}'.format(i)] = kargs['param_val_{}'.format(i)]
                    self['text_{}'.format(i)] = kargs['text_{}'.format(i)]

            elif kargs['conversion_type'] == v23c.CONVERSION_TYPE_RTABX:
                nr = kargs['ref_param_nr']
                self['block_len'] = v23c.CC_COMMON_BLOCK_SIZE + 20 * nr
                self['range_flag'] = kargs.get('range_flag', 0)
                self['min_phy_value'] = kargs.get('min_phy_value', 0)
                self['max_phy_value'] = kargs.get('max_phy_value', 0)
                self['unit'] = kargs.get('unit', ('\0' * 20).encode('latin-1'))
                self['conversion_type'] = v23c.CONVERSION_TYPE_RTABX
                self['ref_param_nr'] = nr

                self['default_lower'] = 0
                self['default_upper'] = 0
                self['default_addr'] = 0
                key = 'default_addr'
                if key in kargs:
                    self.referenced_blocks[key] = TextBlock(text=kargs[key])
                else:
                    self.referenced_blocks[key] = None

                for i in range(nr - 1):
                    self['lower_{}'.format(i)] = kargs['lower_{}'.format(i)]
                    self['upper_{}'.format(i)] = kargs['upper_{}'.format(i)]
                    key = 'text_{}'.format(i)
                    self[key] = 0
                    self.referenced_blocks[key] = TextBlock(text=kargs[key])
            else:
                message = 'Conversion type "{}" not implemented'
                message = message.format(kargs['conversion_type'])
                logger.exception(message)
                raise MdfException(message)

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
        conversion_type = self['conversion_type']

        if conversion_type == v23c.CONVERSION_TYPE_NONE:
            pass

        elif conversion_type == v23c.CONVERSION_TYPE_LINEAR:
            a = self['a']
            b = self['b']
            if (a, b) != (1, 0):
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
            if (P1, P2, P3, P4, P5, P6) != (0, 1, 0, 0, 0, 1):
                X = values
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

        # compute the keys only for Python < 3.6
        if PYVERSION_MAJOR < 36:
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
        if PYVERSION_MAJOR >= 36:
            result = pack(fmt, *self.values())
        else:
            result = pack(fmt, *[self[key] for key in keys])
        return result

    def __str__(self):
        return 'ChannelConversion (referneced blocks: {}, address: {}, fields: {})'.format(
            self.referenced_blocks,
            hex(self.address),
            super(ChannelConversion, self).__str__(),
        )


class ChannelDependency(dict):
    ''' CDBLOCK class derived from *dict*

    Currently the ChannelDependency object can only be created using the
    *stream* and *address* keyword parameters when reading from file

    The keys have the following meaning:

    * id - Block type identifier, always "CD"
    * block_len - Block size of this block in bytes (entire CDBLOCK)
    * dependency_type - Dependency type
    * sd_nr - Total number of signals dependencies (m)
    * for each dependency there is a group of three keys:

        * dg_{n} - Pointer to the data group block (DGBLOCK) of
            signal dependency *n*
        * cg_{n} - Pointer to the channel group block (DGBLOCK) of
            signal dependency *n*
        * ch_{n} - Pointer to the channel block (DGBLOCK) of
            signal dependency *n*

    * there can also be optional keys which decribe dimensions for
        the N-dimensional dependencies:

        * dim_{n} - Optional: size of dimension *n* for N-dimensional
            dependency

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

    def __init__(self, **kargs):
        super(ChannelDependency, self).__init__()

        self.referenced_channels = []

        try:
            stream = kargs['stream']
            self.address = address = kargs['address']
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
                message = 'Expected "CD" block but found "{}"'
                message = message.format(self['id'])
                logger.exception(message)
                raise MdfException(message)

        except KeyError:
            sd_nr = kargs['sd_nr']
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
                    self['dim_{}'.format(i)] = kargs['dim_{}'.format(i)]
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
        if PYVERSION_MAJOR >= 36:
            result = pack(fmt, *self.values())
        else:
            result = pack(fmt, *[self[key] for key in keys])
        return result


class ChannelExtension(dict):
    ''' CEBLOCK class derived from *dict*

    The ChannelExtension object can be created in two modes:

    * using the *stream* and *address* keyword parameters - when reading
        from file
    * using any of the following presented keys - when creating
        a new ChannelExtension

    The first keys are common for all conversion types, and are followed
    by conversion specific keys. The keys have the following meaning:

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
    stream : file handle
        mdf file handle
    address : int
        block address inside mdf file

    Attributes
    ----------
    address : int
        block address inside mdf file

    '''

    def __init__(self, **kargs):
        super(ChannelExtension, self).__init__()

        self.name = self.path = self.comment = ''

        if 'stream' in kargs:
            stream = kargs['stream']
            try:

                (self['id'],
                 self['block_len'],
                 self['type']) = unpack_from(
                    v23c.FMT_SOURCE_COMMON,
                    kargs['raw_bytes'],
                )
                if self['type'] == v23c.SOURCE_ECU:
                    (self['module_nr'],
                     self['module_address'],
                     self['description'],
                     self['ECU_identification'],
                     self['reserved0']) = unpack_from(
                        v23c.FMT_SOURCE_EXTRA_ECU,
                        kargs['raw_bytes'],
                        6,
                    )
                elif self['type'] == v23c.SOURCE_VECTOR:
                    (self['CAN_id'],
                     self['CAN_ch_index'],
                     self['message_name'],
                     self['sender_name'],
                     self['reserved0']) = unpack_from(
                        v23c.FMT_SOURCE_EXTRA_VECTOR,
                        kargs['raw_bytes'],
                        6,
                    )

                self.address = kargs.get('address', 0)
            except KeyError:

                self.address = address = kargs['address']
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
                     self['reserved0']) = unpack(v23c.FMT_SOURCE_EXTRA_ECU, block)
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
                message = 'Expected "CE" block but found "{}"'
                message = message.format(self['id'])
                logger.exception(message)
                raise MdfException(message)

        else:

            self.address = 0
            self['id'] = b'CE'
            self['block_len'] = kargs.get('block_len', v23c.CE_BLOCK_SIZE)
            self['type'] = kargs.get('type', 2)
            if self['type'] == v23c.SOURCE_ECU:
                self['module_nr'] = kargs.get('module_nr', 0)
                self['module_address'] = kargs.get('module_address', 0)
                self['description'] = kargs.get('description', b'\0')
                self['ECU_identification'] = kargs.get(
                    'ECU_identification',
                    b'\0',
                )
                self['reserved0'] = kargs.get('reserved0', b'\0')
            elif self['type'] == v23c.SOURCE_VECTOR:
                self['CAN_id'] = kargs.get('CAN_id', 0)
                self['CAN_ch_index'] = kargs.get('CAN_ch_index', 0)
                self['message_name'] = kargs.get('message_name', b'\0')
                self['sender_name'] = kargs.get('sender_name', b'\0')
                self['reserved0'] = kargs.get('reserved0', b'\0')

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

        if PYVERSION_MAJOR >= 36:
            result = pack(fmt, *self.values())
        else:
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
    ''' CGBLOCK class derived from *dict*

    The ChannelGroup object can be created in two modes:

    * using the *stream* and *address* keyword parameters - when reading
        from file
    * using any of the following presented keys - when creating
        a new ChannelGroup

    The keys have the following meaning:

    * id - Block type identifier, always "CG"
    * block_len - Block size of this block in bytes (entire CGBLOCK)
    * next_cg_addr - Pointer to next channel group block (CGBLOCK) (NIL
        allowed)
    * first_ch_addr - Pointer to first channel block (CNBLOCK) (NIL allowed)
    * comment_addr - Pointer to channel group comment text (TXBLOCK)
        (NIL allowed)
    * record_id - Record ID, i.e. value of the identifier for a record if
        the DGBLOCK defines a number of record IDs > 0
    * ch_nr - Number of channels (redundant information)
    * samples_byte_nr - Size of data record in Bytes (without record ID),
        i.e. size of plain data for a each recorded sample of this channel
        group
    * cycles_nr - Number of records of this type in the data block
        i.e. number of samples for this channel group
    * sample_reduction_addr - only since version 3.3. Pointer to
        first sample reduction block (SRBLOCK) (NIL allowed) Default value: NIL

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

    def __init__(self, **kargs):
        super(ChannelGroup, self).__init__()
        self.comment = ''

        try:

            stream = kargs['stream']
            self.address = address = kargs['address']
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
                message = 'Expected "CG" block but found "{}"'
                raise MdfException(message.format(self['id']))
            if self['comment_addr']:
                self.comment = get_text_v3(
                    address=self['comment_addr'],
                    stream=stream,
                )
        except KeyError:
            self.address = 0
            self['id'] = b'CG'
            self['block_len'] = kargs.get(
                'block_len',
                v23c.CG_PRE_330_BLOCK_SIZE,
            )
            self['next_cg_addr'] = kargs.get('next_cg_addr', 0)
            self['first_ch_addr'] = kargs.get('first_ch_addr', 0)
            self['comment_addr'] = kargs.get('comment_addr', 0)
            self['record_id'] = kargs.get('record_id', 1)
            self['ch_nr'] = kargs.get('ch_nr', 0)
            self['samples_byte_nr'] = kargs.get('samples_byte_nr', 0)
            self['cycles_nr'] = kargs.get('cycles_nr', 0)
            if self['block_len'] == v23c.CG_POST_330_BLOCK_SIZE:
                self['sample_reduction_addr'] = 0

    def __bytes__(self):
        fmt = v23c.FMT_CHANNEL_GROUP
        keys = v23c.KEYS_CHANNEL_GROUP
        if self['block_len'] == v23c.CG_POST_330_BLOCK_SIZE:
            fmt += 'I'
            keys += ('sample_reduction_addr',)
        if PYVERSION_MAJOR >= 36:
            result = pack(fmt, *self.values())
        else:
            result = pack(fmt, *[self[key] for key in keys])
        return result


class DataBlock(dict):
    """Data Block class derived from *dict*

    The DataBlock object can be created in two modes:

    * using the *stream*, *address* and *size* keyword parameters - when
        reading from file
    * using any of the following presented keys - when creating
        a new ChannelGroup

    The keys have the following meaning:

    * data - bytes block

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

    """

    def __init__(self, **kargs):
        super(DataBlock, self).__init__()

        try:
            stream = kargs['stream']
            size = kargs['size']
            self.address = address = kargs['address']
            stream.seek(address)

            self['data'] = stream.read(size)

        except KeyError:
            self.address = 0
            self['data'] = kargs.get('data', b'')

    def __bytes__(self):
        return self['data']


class DataGroup(dict):
    ''' DGBLOCK class derived from *dict*

    The DataGroup object can be created in two modes:

    * using the *stream* and *address* keyword parameters - when reading
        from file
    * using any of the following presented keys - when creating a new DataGroup

    The keys have the following meaning:

    * id - Block type identifier, always "DG"
    * block_len - Block size of this block in bytes (entire DGBLOCK)
    * next_dg_addr - Pointer to next data group block (DGBLOCK) (NIL allowed)
    * first_cg_addr - Pointer to first channel group block (CGBLOCK)
        (NIL allowed)
    * trigger_addr - Pointer to trigger block (TRBLOCK) (NIL allowed)
    * data_block_addr - Pointer to the data block (see separate chapter
        on data storage)
    * cg_nr - Number of channel groups (redundant information)
    * record_id_nr - Number of record IDs in the data block
    * reserved0 - since version 3.2; Reserved

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

    def __init__(self, **kargs):
        super(DataGroup, self).__init__()

        try:
            stream = kargs['stream']
            self.address = address = kargs['address']
            stream.seek(address)
            block = stream.read(v23c.DG_PRE_320_BLOCK_SIZE)

            (self['id'],
             self['block_len'],
             self['next_dg_addr'],
             self['first_cg_addr'],
             self['trigger_addr'],
             self['data_block_addr'],
             self['cg_nr'],
             self['record_id_nr']) = unpack(v23c.FMT_DATA_GROUP_PRE_320, block)

            if self['block_len'] == v23c.DG_POST_320_BLOCK_SIZE:
                self['reserved0'] = stream.read(4)

            if self['id'] != b'DG':
                message = 'Expected "DG" block but found "{}"'
                message = message.format(self['id'])
                logger.exception(message)
                raise MdfException(message)

        except KeyError:
            self.address = 0
            self['id'] = b'DG'
            self['block_len'] = kargs.get(
                'block_len',
                v23c.DG_PRE_320_BLOCK_SIZE,
            )
            self['next_dg_addr'] = kargs.get('next_dg_addr', 0)
            self['first_cg_addr'] = kargs.get('first_cg_addr', 0)
            self['trigger_addr'] = kargs.get('comment_addr', 0)
            self['data_block_addr'] = kargs.get('data_block_addr', 0)
            self['cg_nr'] = kargs.get('cg_nr', 1)
            self['record_id_nr'] = kargs.get('record_id_nr', 0)
            if self['block_len'] == v23c.DG_POST_320_BLOCK_SIZE:
                self['reserved0'] = b'\0\0\0\0'

    def __bytes__(self):
        if self['block_len'] == v23c.DG_POST_320_BLOCK_SIZE:
            fmt = v23c.FMT_DATA_GROUP_POST_320
            keys = v23c.KEYS_DATA_GROUP_POST_320
        else:
            fmt = v23c.FMT_DATA_GROUP_PRE_320
            keys = v23c.KEYS_DATA_GROUP_PRE_320
        if PYVERSION_MAJOR >= 36:
            result = pack(fmt, *self.values())
        else:
            result = pack(fmt, *[self[key] for key in keys])
        return result


class FileIdentificationBlock(dict):
    ''' IDBLOCK class derived from *dict*

    The TriggerBlock object can be created in two modes:

    * using the *stream* and *address* keyword parameters - when reading
        from file
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
    stream : file handle
        mdf file handle
    version : int
        mdf version in case of new file

    Attributes
    ----------
    address : int
        block address inside mdf file; should be 0 always

    '''

    def __init__(self, **kargs):
        super(FileIdentificationBlock, self).__init__()

        self.address = 0
        try:

            stream = kargs['stream']
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
            version = kargs['version']
            self['file_identification'] = 'MDF     '.encode('latin-1')
            self['version_str'] = version.encode('latin-1') + b'\0' * 4
            self['program_identification'] = 'Python  '.encode('latin-1')
            self['byte_order'] = v23c.BYTE_ORDER_INTEL
            self['float_format'] = 0
            self['mdf_version'] = int(version.replace('.', ''))
            self['code_page'] = 0
            self['reserved0'] = b'\0' * 2
            self['reserved1'] = b'\0' * 26
            self['unfinalized_standard_flags'] = 0
            self['unfinalized_custom_flags'] = 0

    def __bytes__(self):
        if PYVERSION_MAJOR >= 36:
            result = pack(v23c.ID_FMT, *self.values())
        else:
            result = pack(v23c.ID_FMT, *[self[key] for key in v23c.ID_KEYS])
        return result


class HeaderBlock(dict):
    ''' HDBLOCK class derived from *dict*

    The TriggerBlock object can be created in two modes:

    * using the *stream* - when reading from file
    * using the classmethod *from_text*

    The keys have the following meaning:

    * id - Block type identifier, always "HD"
    * block_len - Block size of this block in bytes (entire HDBLOCK)
    * first_dg_addr - Pointer to the first data group block (DGBLOCK)
    * comment_addr - Pointer to the measurement file comment text (TXBLOCK)
        (NIL allowed)
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
    stream : file handle
        mdf file handle

    Attributes
    ----------
    address : int
        block address inside mdf file; should be 64 always

    '''

    def __init__(self, **kargs):
        super(HeaderBlock, self).__init__()

        self.address = 64
        self.program = None
        try:

            stream = kargs['stream']
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
             self['organization'],
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
                message = 'Expected "HD" block but found "{}"'
                message = message.format(self['id'])
                logger.exception(message)
                raise MdfException(message)

            if self['program_addr']:
                self.program = ProgramBlock(
                    address=self['program_addr'],
                    stream=stream,
                )

        except KeyError:
            version = kargs.get('version', '3.20')
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
            self['organization'] = (
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

    @property
    def start_time(self):
        """ get the measurement start timestamp

        Returns
        -------
        timestamp : datetime
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
        if PYVERSION_MAJOR >= 36:
            result = pack(fmt, *self.values())
        else:
            result = pack(fmt, *[self[key] for key in keys])
        return result


class ProgramBlock(dict):
    ''' PRBLOCK class derived from *dict*

    The ProgramBlock object can be created in two modes:

    * using the *stream* and *address* keyword parameters - when reading
        from file
    * using any of the following presented keys - when creating
        a new ProgramBlock

    The keys have the following meaning:

    * id - Block type identifier, always "PR"
    * block_len - Block size of this block in bytes (entire PRBLOCK)
    * data - Program-specific data

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

    def __init__(self, **kargs):
        super(ProgramBlock, self).__init__()

        try:
            stream = kargs['stream']
            self.address = address = kargs['address']
            stream.seek(address)

            (self['id'],
             self['block_len']) = unpack('<2sH', stream.read(4))
            self['data'] = stream.read(self['block_len'] - 4)

            if self['id'] != b'PR':
                message = 'Expected "PR" block but found "{}"'
                message = message.format(self['id'])
                logger.exception(message)
                raise MdfException(message)

        except KeyError:
            self['id'] = b'PR'
            self['block_len'] = len(kargs['data']) + 6
            self['data'] = kargs['data']

    def __bytes__(self):
        fmt = v23c.FMT_PROGRAM_BLOCK.format(self['block_len'])
        if PYVERSION_MAJOR >= 36:
            result = pack(fmt, *self.values())
        else:
            result = pack(fmt, *[self[key] for key in v23c.KEYS_PROGRAM_BLOCK])
        return result


class SampleReduction(dict):
    ''' SRBLOCK class derived from *dict*

    Currently the SampleReduction object can only be created by using
    the *stream* and *address* keyword parameters - when reading from file

    The keys have the following meaning:

    * id - Block type identifier, always "SR"
    * block_len - Block size of this block in bytes (entire SRBLOCK)
    * next_sr_addr - Pointer to next sample reduction block (SRBLOCK)
        (NIL allowed)
    * data_block_addr - Pointer to the data block for this sample reduction
    * cycles_nr - Number of reduced samples in the data block.
    * time_interval - Length of time interval [s] used to calculate
        the reduced samples.

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

    def __init__(self, **kargs):
        super(SampleReduction, self).__init__()

        try:
            stream = kargs['stream']
            self.address = address = kargs['address']
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
                message = 'Expected "SR" block but found "{}"'
                message = message.format(self['id'])
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
    ''' TXBLOCK class derived from *dict*

    The ProgramBlock object can be created in two modes:

    * using the *stream* and *address* keyword parameters - when reading
        from file
    * using the classmethod *from_text*

    The keys have the following meaning:

    * id - Block type identifier, always "TX"
    * block_len - Block size of this block in bytes (entire TXBLOCK)
    * text - Text (new line indicated by CR and LF; end of text indicated by 0)

    Parameters
    ----------
    stream : file handle
        mdf file handle
    address : int
        block address inside mdf file
    text : bytes
        bytes for creating a new TextBlock

    Attributes
    ----------
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
        super(TextBlock, self).__init__()
        try:

            stream = kargs['stream']
            self.address = address = kargs['address']
            stream.seek(address)
            (self['id'],
             self['block_len']) = unpack('<2sH', stream.read(4))
            size = self['block_len'] - 4
            self['text'] = stream.read(size)

            if self['id'] != b'TX':
                message = 'Expected "TX" block but found "{}"'
                message = message.format(self['id'])
                logger.exception(message)
                raise MdfException(message)

        except KeyError:
            self.address = 0
            text = kargs['text']

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
        if PYVERSION_MAJOR >= 36:
            result = pack(
                '<2sH{}s'.format(self['block_len'] - 4),
                *self.values()
            )
        else:
            result = pack(
                '<2sH{}s'.format(self['block_len'] - 4),
                *[self[key] for key in v23c.KEYS_TEXT_BLOCK]
            )
        return result


class TriggerBlock(dict):
    ''' TRBLOCK class derived from *dict*

    The TriggerBlock object can be created in two modes:

    * using the *stream* and *address* keyword parameters - when reading
        from file
    * using the classmethod *from_text*

    The keys have the following meaning:

    * id - Block type identifier, always "TR"
    * block_len - Block size of this block in bytes (entire TRBLOCK)
    * text_addr - Pointer to trigger comment text (TXBLOCK) (NIL allowed)
    * trigger_events_nr - Number of trigger events n (0 allowed)
    * trigger_{n}_time - Trigger time [s] of trigger event *n*
    * trigger_{n}_pretime - Pre trigger time [s] of trigger event *n*
    * trigger_{n}_posttime - Post trigger time [s] of trigger event *n*

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

    def __init__(self, **kargs):
        super(TriggerBlock, self).__init__()

        try:
            self.address = address = kargs['address']
            stream = kargs['stream']

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

            if self['id'] != b'TR':
                message = 'Expected "TR" block but found "{}"'
                message = message.format(self['id'])
                logger.exception(message)
                raise MdfException(message)

        except KeyError:
            self.address = 0
            nr = 0
            while 'trigger_{}_time'.format(nr) in kargs:
                nr += 1

            self['id'] = b'TR'
            self['block_len'] = 10 + nr * 8 * 3
            self['text_addr'] = 0
            self['trigger_events_nr'] = nr

            for i in range(nr):
                key = 'trigger_{}_time'.format(i)
                self[key] = kargs[key]
                key = 'trigger_{}_pretime'.format(i)
                self[key] = kargs[key]
                key = 'trigger_{}_posttime'.format(i)
                self[key] = kargs[key]

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
        if PYVERSION_MAJOR >= 36:
            result = pack(fmt, *self.values())
        else:
            result = pack(fmt, *[self[key] for key in keys])
        return result

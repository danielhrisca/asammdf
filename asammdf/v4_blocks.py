# -*- coding: utf-8 -*-
"""
classes that implement the blocks for MDF version 4
"""
from __future__ import division, print_function

import sys
import time
import warnings
from hashlib import md5
from struct import pack, unpack, unpack_from
from zlib import compress, decompress

import numpy as np

from . import v4_constants as v4c
from .utils import MdfException

PYVERSION = sys.version_info[0]
PYVERSION_MAJOR = sys.version_info[0] * 10 + sys.version_info[1]
SEEK_START = v4c.SEEK_START
SEEK_END = v4c.SEEK_END

__all__ = [
    'AttachmentBlock',
    'Channel',
    'ChannelArrayBlock',
    'ChannelGroup',
    'ChannelConversion',
    'DataBlock',
    'DataZippedBlock',
    'FileIdentificationBlock',
    'HeaderBlock',
    'HeaderList',
    'DataList',
    'DataGroup',
    'FileHistory',
    'SignalDataBlock',
    'SourceInformation',
    'TextBlock',
]


class AttachmentBlock(dict):
    """ ATBLOCK class

    When adding new attachments only embedded attachemnts are allowed, with
    keyword argument *data* of type bytes"""
    __slots__ = ['address', ]

    def __init__(self, **kargs):
        super(AttachmentBlock, self).__init__()

        try:
            self.address = address = kargs['address']
            stream = kargs['stream']
            stream.seek(address, SEEK_START)

            (self['id'],
             self['reserved0'],
             self['block_len'],
             self['links_nr'],
             self['next_at_addr'],
             self['file_name_addr'],
             self['mime_addr'],
             self['comment_addr'],
             self['flags'],
             self['creator_index'],
             self['reserved1'],
             self['md5_sum'],
             self['original_size'],
             self['embedded_size']) = unpack(
                v4c.FMT_AT_COMMON,
                stream.read(v4c.AT_COMMON_SIZE),
            )

            self['embedded_data'] = stream.read(self['embedded_size'])

            if self['id'] != b'##AT':
                message = 'Expected "##AT" block but found "{}"'
                raise MdfException(message.format(self['id']))

        except KeyError:

            data = kargs['data']
            size = len(data)
            compression = kargs.get('compression', False)

            if compression:
                data = compress(data)
                original_size = size
                size = len(data)
                self['id'] = b'##AT'
                self['reserved0'] = 0
                self['block_len'] = v4c.AT_COMMON_SIZE + size
                self['links_nr'] = 4
                self['next_at_addr'] = 0
                self['file_name_addr'] = 0
                self['mime_addr'] = 0
                self['comment_addr'] = 0
                self['flags'] = v4c.FLAG_AT_EMBEDDED | v4c.FLAG_AT_MD5_VALID | v4c.FLAG_AT_COMPRESSED_EMBEDDED
                self['creator_index'] = 0
                self['reserved1'] = 0
                md5_worker = md5()
                md5_worker.update(data)
                self['md5_sum'] = md5_worker.digest()
                self['original_size'] = original_size
                self['embedded_size'] = size
                self['embedded_data'] = data
            else:
                self['id'] = b'##AT'
                self['reserved0'] = 0
                self['block_len'] = v4c.AT_COMMON_SIZE + size
                self['links_nr'] = 4
                self['next_at_addr'] = 0
                self['file_name_addr'] = 0
                self['mime_addr'] = 0
                self['comment_addr'] = 0
                self['flags'] = v4c.FLAG_AT_EMBEDDED | v4c.FLAG_AT_MD5_VALID
                self['creator_index'] = 0
                self['reserved1'] = 0
                md5_worker = md5()
                md5_worker.update(data)
                self['md5_sum'] = md5_worker.digest()
                self['original_size'] = size
                self['embedded_size'] = size
                self['embedded_data'] = data

    def extract(self):
        if self['flags'] & v4c.FLAG_AT_EMBEDDED:
            if self['flags'] & v4c.FLAG_AT_COMPRESSED_EMBEDDED:
                data = decompress(self['embedded_data'])
            else:
                data = self['embedded_data']
            if self['flags'] & v4c.FLAG_AT_MD5_VALID:
                md5_worker = md5()
                md5_worker.update(data)
                md5_sum = md5_worker.digest()
                if self['md5_sum'] == md5_sum:
                    return data
                else:
                    message = ('ATBLOCK md5sum="{}"'
                               ' and embedded data md5sum="{}"')
                    warnings.warn(message.format(self['md5_sum'], md5_sum))
        else:
            warnings.warn('external attachments not supported')

    def __bytes__(self):
        fmt = v4c.FMT_AT_COMMON + '{}s'.format(self['embedded_size'])
        if PYVERSION_MAJOR >= 36:
            result = pack(fmt, *self.values())
        else:
            result = pack(fmt, *[self[key] for key in v4c.KEYS_AT_BLOCK])
        return result


class Channel(dict):
    """ CNBLOCK class"""
    __slots__ = ['address', 'name', 'unit', 'comment', 'comment_type']

    def __init__(self, **kargs):
        super(Channel, self).__init__()

        self.name = self.unit = self.comment = self.comment_type = ''

        if 'stream' in kargs:

            self.address = address = kargs['address']
            stream = kargs['stream']
            stream.seek(address, SEEK_START)

            (self['id'],
             self['reserved0'],
             self['block_len'],
             self['links_nr']) = unpack(
                v4c.FMT_COMMON,
                stream.read(v4c.COMMON_SIZE),
            )

            block = stream.read(self['block_len'] - v4c.COMMON_SIZE)

            links_nr = self['links_nr']

            links = unpack_from('<{}Q'.format(links_nr), block)
            params = unpack_from(v4c.FMT_CHANNEL_PARAMS, block, links_nr * 8)

            (self['next_ch_addr'],
             self['component_addr'],
             self['name_addr'],
             self['source_addr'],
             self['conversion_addr'],
             self['data_block_addr'],
             self['unit_addr'],
             self['comment_addr']) = links[:8]

            for i in range(params[10]):
                self['attachment_{}_addr'.format(i)] = links[8 + i]

            if params[6] & v4c.FLAG_CN_DEFAULT_X:
                (self['default_X_dg_addr'],
                 self['default_X_cg_addr'],
                 self['default_X_ch_addr']) = links[-3:]

                # default X not supported yet
                (self['default_X_dg_addr'],
                 self['default_X_cg_addr'],
                 self['default_X_ch_addr']) = (0, 0, 0)

            (self['channel_type'],
             self['sync_type'],
             self['data_type'],
             self['bit_offset'],
             self['byte_offset'],
             self['bit_count'],
             self['flags'],
             self['pos_invalidation_bit'],
             self['precision'],
             self['reserved1'],
             self['attachment_nr'],
             self['min_raw_value'],
             self['max_raw_value'],
             self['lower_limit'],
             self['upper_limit'],
             self['lower_ext_limit'],
             self['upper_ext_limit']) = unpack_from(
                v4c.FMT_CHANNEL_PARAMS,
                block,
                links_nr * 8,
            )

            if self['id'] != b'##CN':
                message = 'Expected "##CN" block but found "{}"'
                raise MdfException(message.format(self['id']))

        else:
            self.address = 0

            self['id'] = b'##CN'
            self['reserved0'] = 0
            self['block_len'] = v4c.CN_BLOCK_SIZE
            self['links_nr'] = 8
            self['next_ch_addr'] = 0
            self['component_addr'] = 0
            self['name_addr'] = kargs.get('name_addr', 0)
            self['source_addr'] = 0
            self['conversion_addr'] = 0
            self['data_block_addr'] = 0
            self['unit_addr'] = kargs.get('unit_addr', 0)
            self['comment_addr'] = 0
            self['channel_type'] = kargs['channel_type']
            self['sync_type'] = kargs.get('sync_type', 0)
            self['data_type'] = kargs['data_type']
            self['bit_offset'] = kargs['bit_offset']
            self['byte_offset'] = kargs['byte_offset']
            self['bit_count'] = kargs['bit_count']
            self['flags'] = kargs.get('flags', 28)
            self['pos_invalidation_bit'] = 0
            self['precision'] = 3
            self['reserved1'] = 0
            self['attachment_nr'] = 0
            self['min_raw_value'] = kargs.get('min_raw_value', 0)
            self['max_raw_value'] = kargs.get('max_raw_value', 0)
            self['lower_limit'] = kargs.get('lower_limit', 0)
            self['upper_limit'] = kargs.get('upper_limit', 100)
            self['lower_ext_limit'] = kargs.get('lower_ext_limit', 0)
            self['upper_ext_limit'] = kargs.get('upper_ext_limit', 0)

        # ignore MLSD signal data
        if self['channel_type'] == v4c.CHANNEL_TYPE_MLSD:
            self['data_block_addr'] = 0
            self['channel_type'] = v4c.CHANNEL_TYPE_VALUE

    def __bytes__(self):

        fmt = v4c.FMT_CHANNEL.format(self['links_nr'])

        if PYVERSION_MAJOR >= 36:
            result = pack(fmt, *self.values())
        else:
            keys = [
                'id',
                'reserved0',
                'block_len',
                'links_nr',
                'next_ch_addr',
                'component_addr',
                'name_addr',
                'source_addr',
                'conversion_addr',
                'data_block_addr',
                'unit_addr',
                'comment_addr',
            ]
            for i in range(self['attachment_nr']):
                keys.append('attachment_{}_addr'.format(i))
            if self['flags'] & v4c.FLAG_CN_DEFAULT_X:
                keys += [
                    'default_X_dg_addr',
                    'default_X_cg_addr',
                    'default_X_ch_addr',
                ]
            keys += [
                'channel_type',
                'sync_type',
                'data_type',
                'bit_offset',
                'byte_offset',
                'bit_count',
                'flags',
                'pos_invalidation_bit',
                'precision',
                'reserved1',
                'attachment_nr',
                'min_raw_value',
                'max_raw_value',
                'lower_limit',
                'upper_limit',
                'lower_ext_limit',
                'upper_ext_limit',
            ]
            result = pack(fmt, *[self[key] for key in keys])
        return result

    def __lt__(self, other):
        self_byte_offset = self['byte_offset']
        other_byte_offset = other['byte_offset']

        if self_byte_offset < other_byte_offset:
            result = 1
        elif self_byte_offset == other_byte_offset:
            self_range = self['bit_offset'] + self['bit_count']
            other_range = other['bit_offset'] + other['bit_count']

            if self_range > other_range:
                result = 1
            else:
                result = 0
        else:
            result = 0
        return result


class ChannelArrayBlock(dict):
    """CABLOCK class"""
    __slots__ = ['address', 'referenced_channels']

    def __init__(self, **kargs):
        super(ChannelArrayBlock, self).__init__()

        self.referenced_channels = []

        try:
            self.address = address = kargs['address']
            stream = kargs['stream']
            stream.seek(address, SEEK_START)

            (self['id'],
             self['reserved0'],
             self['block_len'],
             self['links_nr']) = unpack('<4sI2Q', stream.read(24))

            nr = self['links_nr']
            links = unpack('<{}Q'.format(nr), stream.read(8 * nr))
            self['composition_addr'] = links[0]

            values = unpack('<2BHIiI', stream.read(16))
            dims_nr = values[2]

            if nr == 1:
                pass

            # lookup table with fixed axis
            elif nr == dims_nr + 1:
                for i in range(dims_nr):
                    self['axis_conversion_{}'.format(i)] = links[i + 1]

            # lookup table with CN template
            elif nr == 4 * dims_nr + 1:
                for i in range(dims_nr):
                    self['axis_conversion_{}'.format(i)] = links[i + 1]
                links = links[dims_nr + 1:]
                for i in range(dims_nr):
                    self['scale_axis_{}_dg_addr'.format(i)] = links[3 * i]
                    self['scale_axis_{}_cg_addr'.format(i)] = links[3 * i + 1]
                    self['scale_axis_{}_ch_addr'.format(i)] = links[3 * i + 2]

            (self['ca_type'],
             self['storage'],
             self['dims'],
             self['flags'],
             self['byte_offset_base'],
             self['invalidation_bit_base']) = values

            dim_sizes = unpack('<{}Q'.format(dims_nr), stream.read(8 * dims_nr))
            for i, size in enumerate(dim_sizes):
                self['dim_size_{}'.format(i)] = size

            if self['flags'] & v4c.FLAG_CA_FIXED_AXIS:
                for i in range(dims_nr):
                    for j in range(self['dim_size_{}'.format(i)]):
                        value = unpack('<d', stream.read(8))[0]
                        self['axis_{}_value_{}'.format(i, j)] = value

            if self['id'] != b'##CA':
                message = 'Expected "##CA" block but found "{}"'
                raise MdfException(message.format(self['id']))

        except KeyError:
            self['id'] = b'##CA'
            self['reserved0'] = 0

            ca_type = kargs['ca_type']

            if ca_type == v4c.CA_TYPE_ARRAY:
                dims_nr = kargs['dims']
                self['block_len'] = 48 + dims_nr * 8
                self['links_nr'] = 1
                self['composition_addr'] = 0
                self['ca_type'] = v4c.CA_TYPE_ARRAY
                self['storage'] = v4c.CA_STORAGE_TYPE_CN_TEMPLATE
                self['dims'] = dims_nr
                self['flags'] = 0
                self['byte_offset_base'] = kargs.get('byte_offset_base', 1)
                self['invalidation_bit_base'] = kargs.get(
                    'invalidation_bit_base',
                    0,
                )
                for i in range(dims_nr):
                    self['dim_size_{}'.format(i)] = kargs['dim_size_{}'.format(i)]
            elif ca_type == v4c.CA_TYPE_SCALE_AXIS:
                self['block_len'] = 56
                self['links_nr'] = 1
                self['composition_addr'] = 0
                self['ca_type'] = v4c.CA_TYPE_SCALE_AXIS
                self['storage'] = v4c.CA_STORAGE_TYPE_CN_TEMPLATE
                self['dims'] = 1
                self['flags'] = 0
                self['byte_offset_base'] = kargs.get('byte_offset_base', 1)
                self['invalidation_bit_base'] = kargs.get(
                    'invalidation_bit_base',
                    0,
                )
                self['dim_size_0'] = kargs['dim_size_0']
            elif ca_type == v4c.CA_TYPE_LOOKUP:
                flags = kargs['flags']
                dims_nr = kargs['dims']
                values = sum(kargs['dim_size_{}'.format(i)] for i in range(dims_nr))
                if flags & v4c.FLAG_CA_FIXED_AXIS:
                    self['block_len'] = 48 + dims_nr * 16 + values * 8
                    self['links_nr'] = 1 + dims_nr
                    self['composition_addr'] = 0
                    for i in range(dims_nr):
                        self['axis_conversion_{}'.format(i)] = 0
                    self['ca_type'] = v4c.CA_TYPE_LOOKUP
                    self['storage'] = v4c.CA_STORAGE_TYPE_CN_TEMPLATE
                    self['dims'] = dims_nr
                    self['flags'] = v4c.FLAG_CA_FIXED_AXIS | v4c.FLAG_CA_AXIS
                    self['byte_offset_base'] = kargs.get('byte_offset_base', 1)
                    self['invalidation_bit_base'] = kargs.get(
                        'invalidation_bit_base',
                        0,
                    )
                    for i in range(dims_nr):
                        self['dim_size_{}'.format(i)] = kargs['dim_size_{}'.format(i)]
                    for i in range(dims_nr):
                        for j in range(self['dim_size_{}'.format(i)]):
                            self['axis_{}_value_{}'.format(i, j)] = kargs.get(
                                'axis_{}_value_{}'.format(i, j),
                                j,
                            )
                else:
                    self['block_len'] = 48 + dims_nr * 5 * 8
                    self['links_nr'] = 1 + dims_nr * 4
                    self['composition_addr'] = 0
                    for i in range(dims_nr):
                        self['axis_conversion_{}'.format(i)] = 0
                    for i in range(dims_nr):
                        self['scale_axis_{}_dg_addr'.format(i)] = 0
                        self['scale_axis_{}_cg_addr'.format(i)] = 0
                        self['scale_axis_{}_ch_addr'.format(i)] = 0
                    self['ca_type'] = v4c.CA_TYPE_LOOKUP
                    self['storage'] = v4c.CA_STORAGE_TYPE_CN_TEMPLATE
                    self['dims'] = dims_nr
                    self['flags'] = v4c.FLAG_CA_AXIS
                    self['byte_offset_base'] = kargs.get('byte_offset_base', 1)
                    self['invalidation_bit_base'] = kargs.get(
                        'invalidation_bit_base',
                        0,
                    )
                    for i in range(dims_nr):
                        self['dim_size_{}'.format(i)] = kargs['dim_size_{}'.format(i)]

    def __bytes__(self):
        flags = self['flags']
        ca_type = self['ca_type']
        dims_nr = self['dims']

        if ca_type == v4c.CA_TYPE_ARRAY:
            keys = (
                'id',
                'reserved0',
                'block_len',
                'links_nr',
                'composition_addr',
                'ca_type',
                'storage',
                'dims',
                'flags',
                'byte_offset_base',
                'invalidation_bit_base',
            )
            keys += tuple('dim_size_{}'.format(i) for i in range(dims_nr))
            fmt = '<4sI3Q2BHIiI{}Q'.format(dims_nr)
        elif ca_type == v4c.CA_TYPE_SCALE_AXIS:
            keys = (
                'id',
                'reserved0',
                'block_len',
                'links_nr',
                'composition_addr',
                'ca_type',
                'storage',
                'dims',
                'flags',
                'byte_offset_base',
                'invalidation_bit_base',
                'dim_size_0',
            )
            fmt = '<4sI3Q2BHIiIQ'
        elif ca_type == v4c.CA_TYPE_LOOKUP:
            if flags & v4c.FLAG_CA_FIXED_AXIS:
                nr = sum(self['dim_size_{}'.format(i)] for i in range(dims_nr))
                keys = (
                    'id',
                    'reserved0',
                    'block_len',
                    'links_nr',
                    'composition_addr',
                )
                keys += tuple(
                    'axis_conversion_{}'.format(i)
                    for i in range(dims_nr)
                )
                keys += (
                    'ca_type',
                    'storage',
                    'dims',
                    'flags',
                    'byte_offset_base',
                    'invalidation_bit_base',
                )
                keys += tuple('dim_size_{}'.format(i) for i in range(dims_nr))
                keys += tuple(
                    'axis_{}_value_{}'.format(i, j)
                    for i in range(dims_nr)
                    for j in range(self['dim_size_{}'.format(i)])
                )
                fmt = '<4sI{}Q2BHIiI{}Q{}d'
                fmt = fmt.format(self['links_nr'] + 2, dims_nr, nr)
            else:
                keys = (
                    'id',
                    'reserved0',
                    'block_len',
                    'links_nr',
                    'composition_addr',
                )
                keys += tuple('axis_conversion_{}'.format(i)
                              for i in range(dims_nr))
                for i in range(dims_nr):
                    keys += (
                        'scale_axis_{}_dg_addr'.format(i),
                        'scale_axis_{}_cg_addr'.format(i),
                        'scale_axis_{}_ch_addr'.format(i),
                    )
                keys += (
                    'ca_type',
                    'storage',
                    'dims',
                    'flags',
                    'byte_offset_base',
                    'invalidation_bit_base',
                )
                keys += tuple('dim_size_{}'.format(i) for i in range(dims_nr))
                fmt = '<4sI{}Q2BHIiI{}Q'.format(self['links_nr'] + 2, dims_nr)

        if PYVERSION_MAJOR >= 36:
            result = pack(fmt, *self.values())
        else:
            result = pack(fmt, *[self[key] for key in keys])
        return result


class ChannelGroup(dict):
    """CGBLOCK class"""

    __slots__ = ['address', ]

    def __init__(self, **kargs):
        super(ChannelGroup, self).__init__()

        try:
            self.address = address = kargs['address']
            stream = kargs['stream']
            stream.seek(address, SEEK_START)

            (self['id'],
             self['reserved0'],
             self['block_len'],
             self['links_nr'],
             self['next_cg_addr'],
             self['first_ch_addr'],
             self['acq_name_addr'],
             self['acq_source_addr'],
             self['first_sample_reduction_addr'],
             self['comment_addr'],
             self['record_id'],
             self['cycles_nr'],
             self['flags'],
             self['path_separator'],
             self['reserved1'],
             self['samples_byte_nr'],
             self['invalidation_bytes_nr']) = unpack(
                v4c.FMT_CHANNEL_GROUP,
                stream.read(v4c.CG_BLOCK_SIZE),
            )

            if self['id'] != b'##CG':
                message = 'Expected "##CG" block but found "{}"'
                raise MdfException(message.format(self['id']))

        except KeyError:
            self.address = 0
            self['id'] = b'##CG'
            self['reserved0'] = kargs.get('reserved0', 0)
            self['block_len'] = kargs.get('block_len', v4c.CG_BLOCK_SIZE)
            self['links_nr'] = kargs.get('links_nr', 6)
            self['next_cg_addr'] = kargs.get('next_cg_addr', 0)
            self['first_ch_addr'] = kargs.get('first_ch_addr', 0)
            self['acq_name_addr'] = kargs.get('acq_name_addr', 0)
            self['acq_source_addr'] = kargs.get('acq_source_addr', 0)
            self['first_sample_reduction_addr'] = kargs.get(
                'first_sample_reduction_addr',
                0,
            )
            self['comment_addr'] = kargs.get('comment_addr', 0)
            self['record_id'] = kargs.get('record_id', 1)
            self['cycles_nr'] = kargs.get('cycles_nr', 0)
            self['flags'] = kargs.get('flags', 0)
            self['path_separator'] = kargs.get('path_separator', 0)
            self['reserved1'] = kargs.get('reserved1', 0)
            self['samples_byte_nr'] = kargs.get('samples_byte_nr', 0)
            self['invalidation_bytes_nr'] = kargs.get(
                'invalidation_bytes_nr',
                0,
            )

        # sample reduction blocks are not supported yet
        self['first_sample_reduction_addr'] = 0

    def __bytes__(self):
        if PYVERSION_MAJOR >= 36:
            result = pack(v4c.FMT_CHANNEL_GROUP, *self.values())
        else:
            result = pack(
                v4c.FMT_CHANNEL_GROUP,
                *[self[key] for key in v4c.KEYS_CHANNEL_GROUP]
            )
        return result


class ChannelConversion(dict):
    """CCBLOCK class"""

    __slots__ = ['address', 'name', 'unit', 'comment', 'formula']

    def __init__(self, **kargs):
        super(ChannelConversion, self).__init__()

        self.name = self.unit = self.comment = self.formula = ''

        if 'raw_bytes' in kargs or 'stream' in kargs:
            try:  
                (self['id'],
                 self['reserved0'],
                 self['block_len'],
                 self['links_nr']) = unpack_from(
                    v4c.FMT_COMMON,
                    kargs['raw_bytes'],
                )

                self.address = 0

                block = kargs['raw_bytes'][v4c.COMMON_SIZE:]

            except KeyError:

                self.address = address = kargs['address']
                stream = kargs['stream']
                stream.seek(address, SEEK_START)

                (self['id'],
                self['reserved0'],
                self['block_len'],
                self['links_nr']) = unpack(
                    v4c.FMT_COMMON,
                    stream.read(v4c.COMMON_SIZE),
                )

                block = stream.read(self['block_len'] - v4c.COMMON_SIZE)

            conv = unpack_from('<B', block, self['links_nr'] * 8)[0]

            if conv == v4c.CONVERSION_TYPE_NON:
                (self['name_addr'],
                 self['unit_addr'],
                 self['comment_addr'],
                 self['inv_conv_addr'],
                 self['conversion_type'],
                 self['precision'],
                 self['flags'],
                 self['ref_param_nr'],
                 self['val_param_nr'],
                 self['min_phy_value'],
                 self['max_phy_value']) = unpack(
                    v4c.FMT_CONVERSION_NONE_INIT,
                    block,
                )

            elif conv == v4c.CONVERSION_TYPE_LIN:
                (self['name_addr'],
                 self['unit_addr'],
                 self['comment_addr'],
                 self['inv_conv_addr'],
                 self['conversion_type'],
                 self['precision'],
                 self['flags'],
                 self['ref_param_nr'],
                 self['val_param_nr'],
                 self['min_phy_value'],
                 self['max_phy_value'],
                 self['b'],
                 self['a']) = unpack(v4c.FMT_CONVERSION_LINEAR_INIT, block)

            elif conv == v4c.CONVERSION_TYPE_RAT:
                (self['name_addr'],
                 self['unit_addr'],
                 self['comment_addr'],
                 self['inv_conv_addr'],
                 self['conversion_type'],
                 self['precision'],
                 self['flags'],
                 self['ref_param_nr'],
                 self['val_param_nr'],
                 self['min_phy_value'],
                 self['max_phy_value'],
                 self['P1'],
                 self['P2'],
                 self['P3'],
                 self['P4'],
                 self['P5'],
                 self['P6']) = unpack(v4c.FMT_CONVERSION_RAT_INIT, block)

            elif conv == v4c.CONVERSION_TYPE_ALG:
                (self['name_addr'],
                 self['unit_addr'],
                 self['comment_addr'],
                 self['inv_conv_addr'],
                 self['formula_addr'],
                 self['conversion_type'],
                 self['precision'],
                 self['flags'],
                 self['ref_param_nr'],
                 self['val_param_nr'],
                 self['min_phy_value'],
                 self['max_phy_value']) = unpack(
                    v4c.FMT_CONVERSION_ALGEBRAIC_INIT,
                    block,
                )

            elif conv in (v4c.CONVERSION_TYPE_TABI, v4c.CONVERSION_TYPE_TAB):
                (self['name_addr'],
                 self['unit_addr'],
                 self['comment_addr'],
                 self['inv_conv_addr'],
                 self['conversion_type'],
                 self['precision'],
                 self['flags'],
                 self['ref_param_nr'],
                 self['val_param_nr'],
                 self['min_phy_value'],
                 self['max_phy_value']) = unpack_from(
                    v4c.FMT_CONVERSION_NONE_INIT,
                    block,
                )

                nr = self['val_param_nr']
                values = unpack('<{}d'.format(nr), block[56:])
                for i in range(nr // 2):
                    (self['raw_{}'.format(i)],
                     self['phys_{}'.format(i)]) = values[i * 2], values[2 * i + 1]

            elif conv == v4c.CONVERSION_TYPE_RTAB:
                (self['name_addr'],
                 self['unit_addr'],
                 self['comment_addr'],
                 self['inv_conv_addr'],
                 self['conversion_type'],
                 self['precision'],
                 self['flags'],
                 self['ref_param_nr'],
                 self['val_param_nr'],
                 self['min_phy_value'],
                 self['max_phy_value']) = unpack_from(
                    v4c.FMT_CONVERSION_NONE_INIT,
                    block,
                )
                nr = self['val_param_nr']
                values = unpack('<{}d'.format(nr), block[56:])
                for i in range((nr - 1) // 3):
                    (self['lower_{}'.format(i)],
                     self['upper_{}'.format(i)],
                     self['phys_{}'.format(i)]) = values[i * 3], values[3 * i + 1], values[3 * i + 2]
                self['default'] = unpack('<d', block[-8:])[0]

            elif conv == v4c.CONVERSION_TYPE_TABX:
                (self['name_addr'],
                 self['unit_addr'],
                 self['comment_addr'],
                 self['inv_conv_addr']) = unpack_from('<4Q', block)

                links_nr = self['links_nr'] - 4

                links = unpack_from('<{}Q'.format(links_nr), block, 32)
                for i, link in enumerate(links[:-1]):
                    self['text_{}'.format(i)] = link
                self['default_addr'] = links[-1]

                (self['conversion_type'],
                 self['precision'],
                 self['flags'],
                 self['ref_param_nr'],
                 self['val_param_nr'],
                 self['min_phy_value'],
                 self['max_phy_value']) = unpack_from(
                    '<2B3H2d', block,
                    32 + links_nr * 8,
                )

                values = unpack_from(
                    '<{}d'.format(links_nr - 1),
                    block,
                    32 + links_nr * 8 + 24,
                )
                for i, val in enumerate(values):
                    self['val_{}'.format(i)] = val

            elif conv == v4c.CONVERSION_TYPE_RTABX:
                (self['name_addr'],
                 self['unit_addr'],
                 self['comment_addr'],
                 self['inv_conv_addr']) = unpack_from('<4Q', block)

                links_nr = self['links_nr'] - 4

                links = unpack_from('<{}Q'.format(links_nr), block, 32)
                for i, link in enumerate(links[:-1]):
                    self['text_{}'.format(i)] = link
                self['default_addr'] = links[-1]

                (self['conversion_type'],
                 self['precision'],
                 self['flags'],
                 self['ref_param_nr'],
                 self['val_param_nr'],
                 self['min_phy_value'],
                 self['max_phy_value']) = unpack_from(
                    '<2B3H2d', block,
                    32 + links_nr * 8,
                )

                values = unpack_from(
                    '<{}d'.format((links_nr - 1) * 2),
                    block,
                    32 + links_nr * 8 + 24,
                )
                for i in range(self['val_param_nr'] // 2):
                    j = 2 * i
                    self['lower_{}'.format(i)] = values[j]
                    self['upper_{}'.format(i)] = values[j + 1]

            elif conv == v4c.CONVERSION_TYPE_TTAB:
                (self['name_addr'],
                 self['unit_addr'],
                 self['comment_addr'],
                 self['inv_conv_addr']) = unpack_from('<4Q', block)

                links_nr = self['links_nr'] - 4

                links = unpack_from('<{}Q'.format(links_nr), block, 32)
                for i, link in enumerate(links):
                    self['text_{}'.format(i)] = link

                (self['conversion_type'],
                 self['precision'],
                 self['flags'],
                 self['ref_param_nr'],
                 self['val_param_nr'],
                 self['min_phy_value'],
                 self['max_phy_value']) = unpack_from(
                    '<2B3H2d', block,
                    32 + links_nr * 8,
                )

                values = unpack_from(
                    '<{}d'.format(self['val_param_nr']),
                    block,
                    32 + links_nr * 8 + 24,
                )
                for i, val in enumerate(values[:-1]):
                    self['val_{}'.format(i)] = val
                self['val_default'] = values[-1]

            elif conv == v4c.CONVERSION_TYPE_TRANS:
                (self['name_addr'],
                 self['unit_addr'],
                 self['comment_addr'],
                 self['inv_conv_addr']) = unpack_from('<4Q', block)

                links_nr = self['links_nr'] - 4

                links = unpack_from('<{}Q'.format(links_nr), block, 32)

                for i in range((links_nr - 1) // 2):
                    j = 2 * i
                    self['input_{}_addr'.format(i)] = links[j]
                    self['output_{}_addr'.format(i)] = links[j + 1]
                self['default_addr'] = links[-1]

                (self['conversion_type'],
                 self['precision'],
                 self['flags'],
                 self['ref_param_nr'],
                 self['val_param_nr'],
                 self['min_phy_value'],
                 self['max_phy_value']) = unpack_from(
                    '<2B3H2d', block,
                    32 + links_nr * 8,
                )

            if self['id'] != b'##CC':
                message = 'Expected "##CC" block but found "{}"'
                raise MdfException(message.format(self['id']))

        else:

            self.address = 0
            self['id'] = b'##CC'
            self['reserved0'] = 0

            if kargs['conversion_type'] == v4c.CONVERSION_TYPE_NON:
                self['block_len'] = v4c.CC_NONE_BLOCK_SIZE
                self['links_nr'] = 4
                self['name_addr'] = 0
                self['unit_addr'] = 0
                self['comment_addr'] = 0
                self['inv_conv_addr'] = 0
                self['conversion_type'] = v4c.CONVERSION_TYPE_NON
                self['precision'] = 1
                self['flags'] = 0
                self['ref_param_nr'] = 0
                self['val_param_nr'] = 0
                self['min_phy_value'] = kargs.get('min_phy_value', 0)
                self['max_phy_value'] = kargs.get('max_phy_value', 0)

            elif kargs['conversion_type'] == v4c.CONVERSION_TYPE_LIN:
                self['block_len'] = v4c.CC_LIN_BLOCK_SIZE
                self['links_nr'] = 4
                self['name_addr'] = kargs.get('name_addr', 0)
                self['unit_addr'] = kargs.get('unit_addr', 0)
                self['comment_addr'] = kargs.get('comment_addr', 0)
                self['inv_conv_addr'] = kargs.get('inv_conv_addr', 0)
                self['conversion_type'] = v4c.CONVERSION_TYPE_LIN
                self['precision'] = kargs.get('precision', 1)
                self['flags'] = kargs.get('flags', 0)
                self['ref_param_nr'] = 0
                self['val_param_nr'] = 2
                self['min_phy_value'] = kargs.get('min_phy_value', 0)
                self['max_phy_value'] = kargs.get('max_phy_value', 0)
                self['b'] = kargs['b']
                self['a'] = kargs['a']

            elif kargs['conversion_type'] == v4c.CONVERSION_TYPE_ALG:
                self['block_len'] = kargs.get(
                    'block_len',
                    v4c.CC_ALG_BLOCK_SIZE,
                )
                self['links_nr'] = kargs.get('links_nr', 5)
                self['name_addr'] = kargs.get('name_addr', 0)
                self['unit_addr'] = kargs.get('unit_addr', 0)
                self['comment_addr'] = kargs.get('comment_addr', 0)
                self['inv_conv_addr'] = kargs.get('inv_conv_addr', 0)
                self['conv_text_addr'] = kargs.get('conv_text_addr', 0)
                self['conversion_type'] = v4c.CONVERSION_TYPE_ALG
                self['precision'] = kargs.get('precision', 1)
                self['flags'] = kargs.get('flags', 0)
                self['ref_param_nr'] = kargs.get('ref_param_nr', 1)
                self['val_param_nr'] = kargs.get('val_param_nr', 0)
                self['min_phy_value'] = kargs.get('min_phy_value', 0)
                self['max_phy_value'] = kargs.get('max_phy_value', 0)

            elif kargs['conversion_type'] == v4c.CONVERSION_TYPE_TABX:
                self['block_len'] = ((kargs['links_nr'] - 5) * 8 * 2) + 88
                self['links_nr'] = kargs['links_nr']
                self['name_addr'] = kargs.get('name_addr', 0)
                self['unit_addr'] = kargs.get('unit_addr', 0)
                self['comment_addr'] = kargs.get('comment_addr', 0)
                self['inv_conv_addr'] = kargs.get('inv_conv_addr', 0)
                for i in range(kargs['links_nr'] - 5):
                    self['text_{}'.format(i)] = 0
                self['default_addr'] = kargs.get('default_addr', 0)
                self['conversion_type'] = v4c.CONVERSION_TYPE_TABX
                self['precision'] = kargs.get('precision', 0)
                self['flags'] = kargs.get('flags', 0)
                self['ref_param_nr'] = kargs.get(
                    'ref_param_nr',
                    kargs['links_nr'] - 4,
                )
                self['val_param_nr'] = kargs.get(
                    'val_param_nr',
                    kargs['links_nr'] - 5,
                )
                self['min_phy_value'] = kargs.get('min_phy_value', 0)
                self['max_phy_value'] = kargs.get('max_phy_value', 0)
                for i in range(kargs['links_nr'] - 5):
                    self['val_{}'.format(i)] = kargs['val_{}'.format(i)]

            elif kargs['conversion_type'] == v4c.CONVERSION_TYPE_RTABX:
                self['block_len'] = ((kargs['links_nr'] - 5) * 8 * 3) + 88
                self['links_nr'] = kargs['links_nr']
                self['name_addr'] = kargs.get('name_addr', 0)
                self['unit_addr'] = kargs.get('unit_addr', 0)
                self['comment_addr'] = kargs.get('comment_addr', 0)
                self['inv_conv_addr'] = kargs.get('inv_conv_addr', 0)
                for i in range(kargs['links_nr'] - 5):
                    self['text_{}'.format(i)] = 0
                self['default_addr'] = kargs.get('default_addr', 0)
                self['conversion_type'] = v4c.CONVERSION_TYPE_RTABX
                self['precision'] = kargs.get('precision', 0)
                self['flags'] = kargs.get('flags', 0)
                self['ref_param_nr'] = kargs['links_nr'] - 4
                self['val_param_nr'] = (kargs['links_nr'] - 5) * 2
                self['min_phy_value'] = kargs.get('min_phy_value', 0)
                self['max_phy_value'] = kargs.get('max_phy_value', 0)
                for i in range(kargs['links_nr'] - 5):
                    self['lower_{}'.format(i)] = kargs['lower_{}'.format(i)]
                    self['upper_{}'.format(i)] = kargs['upper_{}'.format(i)]

            elif kargs['conversion_type'] == v4c.CONVERSION_TYPE_TTAB:
                self['block_len'] = ((kargs['links_nr'] - 4) * 8 * 2) + 88
                self['links_nr'] = kargs['links_nr']
                self['name_addr'] = kargs.get('name_addr', 0)
                self['unit_addr'] = kargs.get('unit_addr', 0)
                self['comment_addr'] = kargs.get('comment_addr', 0)
                self['inv_conv_addr'] = kargs.get('inv_conv_addr', 0)
                for i in range(kargs['links_nr'] - 4):
                    self['text_{}'.format(i)] = kargs.get(
                        'text_{}'.format(i),
                        0,
                    )
                self['conversion_type'] = v4c.CONVERSION_TYPE_TTAB
                self['precision'] = kargs.get('precision', 0)
                self['flags'] = kargs.get('flags', 0)
                self['ref_param_nr'] = kargs['links_nr'] - 4
                self['val_param_nr'] = kargs['links_nr'] - 4 + 1
                self['min_phy_value'] = kargs.get('min_phy_value', 0)
                self['max_phy_value'] = kargs.get('max_phy_value', 0)
                for i in range(kargs['links_nr'] - 4):
                    self['val_{}'.format(i)] = kargs['val_{}'.format(i)]
                self['val_default'] = kargs['val_default']

    def __bytes__(self):
        fmt = '<4sI{}Q2B3H{}d'.format(
            self['links_nr'] + 2,
            self['val_param_nr'] + 2,
        )

        # only compute keys for Python < 3.6
        if PYVERSION_MAJOR < 36:
            if self['conversion_type'] == v4c.CONVERSION_TYPE_NON:
                keys = v4c.KEYS_CONVERSION_NONE
            elif self['conversion_type'] == v4c.CONVERSION_TYPE_LIN:
                keys = v4c.KEYS_CONVERSION_LINEAR
            elif self['conversion_type'] == v4c.CONVERSION_TYPE_RAT:
                keys = v4c.KEYS_CONVERSION_RAT
            elif self['conversion_type'] == v4c.CONVERSION_TYPE_ALG:
                keys = v4c.KEYS_CONVERSION_ALGEBRAIC
            elif self['conversion_type'] in (
                    v4c.CONVERSION_TYPE_TABI,
                    v4c.CONVERSION_TYPE_TAB):
                keys = v4c.KEYS_CONVERSION_NONE
                for i in range(self['val_param_nr'] // 2):
                    keys += ('raw_{}'.format(i), 'phys_{}'.format(i))
            elif self['conversion_type'] == v4c.CONVERSION_TYPE_RTAB:
                keys = v4c.KEYS_CONVERSION_NONE
                for i in range(self['val_param_nr'] // 3):
                    keys += (
                        'lower_{}'.format(i),
                        'upper_{}'.format(i),
                        'phys_{}'.format(i),
                    )
                keys += ('default',)
            elif self['conversion_type'] == v4c.CONVERSION_TYPE_TABX:
                keys = (
                    'id',
                    'reserved0',
                    'block_len',
                    'links_nr',
                    'name_addr',
                    'unit_addr',
                    'comment_addr',
                    'inv_conv_addr',
                )
                keys += tuple(
                    'text_{}'.format(i)
                    for i in range(self['links_nr'] - 4 - 1)
                )
                keys += ('default_addr',)
                keys += (
                    'conversion_type',
                    'precision',
                    'flags',
                    'ref_param_nr',
                    'val_param_nr',
                    'min_phy_value',
                    'max_phy_value',
                )
                keys += tuple(
                    'val_{}'.format(i)
                    for i in range(self['val_param_nr'])
                )
            elif self['conversion_type'] == v4c.CONVERSION_TYPE_RTABX:
                keys = (
                    'id',
                    'reserved0',
                    'block_len',
                    'links_nr',
                    'name_addr',
                    'unit_addr',
                    'comment_addr',
                    'inv_conv_addr',
                )
                keys += tuple(
                    'text_{}'.format(i)
                    for i in range(self['links_nr'] - 4 - 1)
                )
                keys += ('default_addr',)
                keys += (
                    'conversion_type',
                    'precision',
                    'flags',
                    'ref_param_nr',
                    'val_param_nr',
                    'min_phy_value',
                    'max_phy_value',
                )
                for i in range(self['val_param_nr'] // 2):
                    keys += (
                        'lower_{}'.format(i),
                        'upper_{}'.format(i),
                    )
            elif self['conversion_type'] == v4c.CONVERSION_TYPE_TTAB:
                keys = (
                    'id',
                    'reserved0',
                    'block_len',
                    'links_nr',
                    'name_addr',
                    'unit_addr',
                    'comment_addr',
                    'inv_conv_addr',
                )
                keys += tuple(
                    'text_{}'.format(i)
                    for i in range(self['links_nr'] - 4)
                )
                keys += (
                    'conversion_type',
                    'precision',
                    'flags',
                    'ref_param_nr',
                    'val_param_nr',
                    'min_phy_value',
                    'max_phy_value',
                )
                keys += tuple(
                    'val_{}'.format(i)
                    for i in range(self['val_param_nr'] - 1)
                )
                keys += ('val_default',)
            elif self['conversion_type'] == v4c.CONVERSION_TYPE_TRANS:
                keys = (
                    'id',
                    'reserved0',
                    'block_len',
                    'links_nr',
                    'name_addr',
                    'unit_addr',
                    'comment_addr',
                    'inv_conv_addr',
                )
                for i in range((self['links_nr'] - 4 - 1) // 2):
                    keys += (
                        'input_{}_addr'.format(i),
                        'output_{}_addr'.format(i),
                    )
                keys += (
                    'default_addr',
                    'conversion_type',
                    'precision',
                    'flags',
                    'ref_param_nr',
                    'val_param_nr',
                    'min_phy_value',
                    'max_phy_value',
                )
                keys += tuple(
                    'val_{}'.format(i)
                    for i in range(self['val_param_nr'] - 1)
                )

        if PYVERSION_MAJOR >= 36:
            result = pack(fmt, *self.values())
        else:
            result = pack(fmt, *[self[key] for key in keys])
        return result


class DataBlock(dict):
    """DTBLOCK class

    Parameters
    ----------
    address : int
        DTBLOCK address inside the file
    stream : int
        file handle

    """
    __slots__ = ['address', ]

    def __init__(self, **kargs):
        super(DataBlock, self).__init__()

        try:
            self.address = address = kargs['address']
            stream = kargs['stream']
            stream.seek(address, SEEK_START)

            (self['id'],
             self['reserved0'],
             self['block_len'],
             self['links_nr']) = unpack(
                v4c.FMT_COMMON,
                stream.read(v4c.COMMON_SIZE),
            )
            self['data'] = stream.read(self['block_len'] - v4c.COMMON_SIZE)

            if self['id'] != b'##DT':
                message = 'Expected "##DT" block but found "{}"'
                raise MdfException(message.format(self['id']))

        except KeyError:

            self['id'] = b'##DT'
            self['reserved0'] = 0
            self['block_len'] = len(kargs['data']) + v4c.COMMON_SIZE
            self['links_nr'] = 0
            self['data'] = kargs['data']

        if PYVERSION_MAJOR < 30 and isinstance(self['data'], bytearray):
            self['data'] = str(self['data'])

    def __bytes__(self):
        fmt = v4c.FMT_DATA_BLOCK.format(self['block_len'] - v4c.COMMON_SIZE)
        if PYVERSION_MAJOR >= 36:
            result = pack(fmt, *self.values())
        else:
            result = pack(fmt, *[self[key] for key in v4c.KEYS_DATA_BLOCK])
        return result


class DataZippedBlock(dict):
    """DZBLOCK class

    Parameters
    ----------
    address : int
        DTBLOCK address inside the file
    stream : int
        file handle

    """
    __slots__ = ['address', 'prevent_data_setitem', 'return_unzipped']

    def __init__(self, **kargs):
        super(DataZippedBlock, self).__init__()

        self.prevent_data_setitem = True
        try:
            self.address = address = kargs['address']
            stream = kargs['stream']
            stream.seek(address, SEEK_START)

            (self['id'],
             self['reserved0'],
             self['block_len'],
             self['links_nr'],
             self['original_type'],
             self['zip_type'],
             self['reserved1'],
             self['param'],
             self['original_size'],
             self['zip_size'],) = unpack(
                v4c.FMT_DZ_COMMON,
                stream.read(v4c.DZ_COMMON_SIZE),
            )

            self['data'] = stream.read(self['zip_size'])

            if self['id'] != b'##DZ':
                message = 'Expected "##DZ" block but found "{}"'
                raise MdfException(message.format(self['id']))

        except KeyError:
            self.prevent_data_setitem = False
            self.address = 0

            data = kargs['data']

            self['id'] = b'##DZ'
            self['reserved0'] = 0
            self['block_len'] = 0
            self['links_nr'] = 0
            self['original_type'] = kargs.get('original_type', b'DT')
            self['zip_type'] = kargs.get('zip_type', v4c.FLAG_DZ_DEFLATE)
            self['reserved1'] = 0
            if self['zip_type'] == v4c.FLAG_DZ_DEFLATE:
                self['param'] = 0
            else:
                self['param'] = kargs['param']

            # since prevent_data_setitem is False the rest of the keys will be
            # handled by __setitem__
            self['data'] = data

        self.prevent_data_setitem = False
        self.return_unzipped = True

    def __setitem__(self, item, value):
        if item == 'data' and self.prevent_data_setitem == False:
            data = value
            self['original_size'] = len(data)

            if self['zip_type'] == v4c.FLAG_DZ_DEFLATE:
                data = compress(data)
            else:
                cols = self['param']
                lines = self['original_size'] // cols

                nd = np.fromstring(data[:lines * cols], dtype=np.uint8)
                nd = nd.reshape((lines, cols))
                data = nd.transpose().tostring() + data[lines * cols:]

                data = compress(data)

            self['zip_size'] = len(data)
            self['block_len'] = self['zip_size'] + v4c.DZ_COMMON_SIZE
            super(DataZippedBlock, self).__setitem__(item, data)
        else:
            super(DataZippedBlock, self).__setitem__(item, value)

    def __getitem__(self, item):
        if item == 'data':
            if self.return_unzipped:
                data = super(DataZippedBlock, self).__getitem__(item)
                data = decompress(data)
                if self['zip_type'] == v4c.FLAG_DZ_TRANPOSED_DEFLATE:
                    cols = self['param']
                    lines = self['original_size'] // cols

                    nd = np.fromstring(data[:lines * cols], dtype=np.uint8)
                    nd = nd.reshape((cols, lines))
                    data = nd.transpose().tostring() + data[lines * cols:]
            else:
                data = super(DataZippedBlock, self).__getitem__(item)
            value = data
        else:
            value = super(DataZippedBlock, self).__getitem__(item)
        return value

    def __bytes__(self):
        fmt = v4c.FMT_DZ_COMMON + '{}s'.format(self['zip_size'])
        self.return_unzipped = False
        if PYVERSION_MAJOR >= 36:
            data = pack(fmt, *self.values())
        else:
            data = pack(fmt, *[self[key] for key in v4c.KEYS_DZ_BLOCK])
        self.return_unzipped = True
        return data


class DataGroup(dict):
    """DGBLOCK class"""
    __slots__ = ['address', ]

    def __init__(self, **kargs):
        super(DataGroup, self).__init__()

        try:
            self.address = address = kargs['address']
            stream = kargs['stream']
            stream.seek(address, SEEK_START)

            (self['id'],
             self['reserved0'],
             self['block_len'],
             self['links_nr'],
             self['next_dg_addr'],
             self['first_cg_addr'],
             self['data_block_addr'],
             self['comment_addr'],
             self['record_id_len'],
             self['reserved1']) = unpack(
                v4c.FMT_DATA_GROUP,
                stream.read(v4c.DG_BLOCK_SIZE),
            )

            if self['id'] != b'##DG':
                message = 'Expected "##DG" block but found "{}"'
                raise MdfException(message.format(self['id']))

        except KeyError:

            self.address = 0
            self['id'] = b'##DG'
            self['reserved0'] = kargs.get('reserved0', 0)
            self['block_len'] = kargs.get('block_len', v4c.DG_BLOCK_SIZE)
            self['links_nr'] = kargs.get('links_nr', 4)
            self['next_dg_addr'] = kargs.get('next_dg_addr', 0)
            self['first_cg_addr'] = kargs.get('first_cg_addr', 0)
            self['data_block_addr'] = kargs.get('data_block_addr', 0)
            self['comment_addr'] = kargs.get('comment_addr', 0)
            self['record_id_len'] = kargs.get('record_id_len', 0)
            self['reserved1'] = kargs.get('reserved1', b'\00' * 7)

    def __bytes__(self):
        if PYVERSION_MAJOR >= 36:
            result = pack(v4c.FMT_DATA_GROUP, *self.values())
        else:
            result = pack(
                v4c.FMT_DATA_GROUP,
                *[self[key] for key in v4c.KEYS_DATA_GROUP]
            )
        return result


class DataList(dict):
    """DLBLOCK class"""
    __slots__ = ['address', ]

    def __init__(self, **kargs):
        super(DataList, self).__init__()

        try:
            self.address = address = kargs['address']
            stream = kargs['stream']
            stream.seek(address, SEEK_START)

            (self['id'],
             self['reserved0'],
             self['block_len'],
             self['links_nr']) = unpack(
                v4c.FMT_COMMON,
                stream.read(v4c.COMMON_SIZE),
            )

            self['next_dl_addr'] = unpack('<Q', stream.read(8))[0]

            links = unpack(
                '<{}Q'.format(self['links_nr'] - 1),
                stream.read((self['links_nr'] - 1) * 8),
            )

            for i, addr in enumerate(links):
                self['data_block_addr{}'.format(i)] = addr

            self['flags'] = stream.read(1)[0]
            if PYVERSION == 2:
                self['flags'] = ord(self['flags'])
            if self['flags'] & v4c.FLAG_DL_EQUAL_LENGHT:
                (self['reserved1'],
                 self['data_block_nr'],
                 self['data_block_len']) = unpack('<3sIQ', stream.read(15))
            else:
                (self['reserved1'],
                 self['data_block_nr']) = unpack('<3sI', stream.read(7))
                offsets = unpack(
                    '<{}Q'.format(self['links_nr'] - 1),
                    stream.read((self['links_nr'] - 1) * 8),
                )
                for i, offset in enumerate(offsets):
                    self['offset_{}'.format(i)] = offset

            if self['id'] != b'##DL':
                message = 'Expected "##DL" block but found "{}"'
                raise MdfException(message.format(self['id']))

        except KeyError:

            self.address = 0
            self['id'] = b'##DL'
            self['reserved0'] = 0
            self['block_len'] = 40 + 8 * kargs.get('links_nr', 2)
            self['links_nr'] = kargs.get('links_nr', 2)
            self['next_dl_addr'] = 0

            for i in range(self['links_nr'] - 1):
                self['data_block_addr{}'.format(i)] = kargs.get(
                    'data_block_addr{}'.format(i),
                    0,
                )

            self['flags'] = kargs.get('flags', 1)
            self['reserved1'] = kargs.get('reserved1', b'\0\0\0')
            self['data_block_nr'] = kargs.get('data_block_nr', 1)
            if self['flags'] & v4c.FLAG_DL_EQUAL_LENGHT:
                self['data_block_len'] = kargs['data_block_len']
            else:
                for i, offset in enumerate(self['links_nr'] - 1):
                    self['offset_{}'.format(i)] = kargs['offset_{}'.format(i)]

    def __bytes__(self):
        fmt = v4c.FMT_DATA_LIST.format(self['links_nr'])
        if PYVERSION_MAJOR < 36:
            keys = (
                'id',
                'reserved0',
                'block_len',
                'links_nr',
                'next_dl_addr',
            )
            keys += tuple(
                'data_block_addr{}'.format(i)
                for i in range(self['links_nr'] - 1)
            )
            keys += (
                'flags',
                'reserved1',
                'data_block_nr',
                'data_block_len',
            )
        if PYVERSION_MAJOR >= 36:
            result = pack(fmt, *self.values())
        else:
            result = pack(fmt, *[self[key] for key in keys])
        return result


class FileIdentificationBlock(dict):
    """IDBLOCK class"""
    __slots__ = ['address', ]

    def __init__(self, **kargs):

        super(FileIdentificationBlock, self).__init__()

        self.address = 0

        try:

            stream = kargs['stream']
            stream.seek(self.address, SEEK_START)

            (self['file_identification'],
             self['version_str'],
             self['program_identification'],
             self['reserved0'],
             self['reserved1'],
             self['mdf_version'],
             self['reserved2'],
             self['check_block'],
             self['fill'],
             self['unfinalized_standard_flags'],
             self['unfinalized_custom_flags']) = unpack(
                v4c.FMT_IDENTIFICATION_BLOCK,
                stream.read(v4c.IDENTIFICATION_BLOCK_SIZE),
            )

        except KeyError:

            version = kargs.get('version', 400)
            self['file_identification'] = 'MDF     '.encode('utf-8')
            self['version_str'] = '{}    '.format(version).encode('utf-8')
            self['program_identification'] = 'Python  '.encode('utf-8')
            self['reserved0'] = 0
            self['reserved1'] = 0
            self['mdf_version'] = int(version.replace('.', ''))
            self['reserved2'] = 0
            self['check_block'] = 0
            self['fill'] = b'\x00' * 26
            self['unfinalized_standard_flags'] = 0
            self['unfinalized_custom_flags'] = 0

    def __bytes__(self):
        if PYVERSION_MAJOR >= 36:
            result = pack(v4c.FMT_IDENTIFICATION_BLOCK, *self.values())
        else:
            result = pack(
                v4c.FMT_IDENTIFICATION_BLOCK,
                *[self[key] for key in v4c.KEYS_IDENTIFICATION_BLOCK]
            )
        return result


class FileHistory(dict):
    """FHBLOCK class"""
    __slots__ = ['address', ]

    def __init__(self, **kargs):
        super(FileHistory, self).__init__()

        try:
            self.address = address = kargs['address']
            stream = kargs['stream']
            stream.seek(address, SEEK_START)

            (self['id'],
             self['reserved0'],
             self['block_len'],
             self['links_nr'],
             self['next_fh_addr'],
             self['comment_addr'],
             self['abs_time'],
             self['tz_offset'],
             self['daylight_save_time'],
             self['time_flags'],
             self['reserved1']) = unpack(
                v4c.FMT_FILE_HISTORY,
                stream.read(v4c.FH_BLOCK_SIZE),
            )

            if self['id'] != b'##FH':
                message = 'Expected "##FH" block but found "{}"'
                raise MdfException(message.format(self['id']))

        except KeyError:
            self['id'] = b'##FH'
            self['reserved0'] = kargs.get('reserved0', 0)
            self['block_len'] = kargs.get('block_len', v4c.FH_BLOCK_SIZE)
            self['links_nr'] = kargs.get('links_nr', 2)
            self['next_fh_addr'] = kargs.get('next_fh_addr', 0)
            self['comment_addr'] = kargs.get('comment_addr', 0)
            self['abs_time'] = kargs.get('abs_time', int(time.time()) * 10 ** 9)
            self['tz_offset'] = kargs.get('tz_offset', 120)
            self['daylight_save_time'] = kargs.get('daylight_save_time', 60)
            self['time_flags'] = kargs.get('time_flags', 2)
            self['reserved1'] = kargs.get('reserved1', b'\x00' * 3)

    def __bytes__(self):
        if PYVERSION_MAJOR >= 36:
            result = pack(v4c.FMT_FILE_HISTORY, *self.values())
        else:
            result = pack(
                v4c.FMT_FILE_HISTORY,
                *[self[key] for key in v4c.KEYS_FILE_HISTORY]
            )
        return result


class HeaderBlock(dict):
    """HDBLOCK class"""
    __slots__ = ['address', ]

    def __init__(self, **kargs):
        super(HeaderBlock, self).__init__()

        try:
            self.address = address = kargs['address']
            stream = kargs['stream']
            stream.seek(address, SEEK_START)

            (self['id'],
             self['reserved3'],
             self['block_len'],
             self['links_nr'],
             self['first_dg_addr'],
             self['file_history_addr'],
             self['channel_tree_addr'],
             self['first_attachment_addr'],
             self['first_event_addr'],
             self['comment_addr'],
             self['abs_time'],
             self['tz_offset'],
             self['daylight_save_time'],
             self['time_flags'],
             self['time_quality'],
             self['flags'],
             self['reserved4'],
             self['start_angle'],
             self['start_distance']) = unpack(
                v4c.FMT_HEADER_BLOCK,
                stream.read(v4c.HEADER_BLOCK_SIZE),
            )

            if self['id'] != b'##HD':
                message = 'Expected "##HD" block but found "{}"'
                raise MdfException(message.format(self['id']))

        except KeyError:

            self['id'] = b'##HD'
            self['reserved3'] = kargs.get('reserved3', 0)
            self['block_len'] = kargs.get('block_len', v4c.HEADER_BLOCK_SIZE)
            self['links_nr'] = kargs.get('links_nr', 6)
            self['first_dg_addr'] = kargs.get('first_dg_addr', 0)
            self['file_history_addr'] = kargs.get('file_history_addr', 0)
            self['channel_tree_addr'] = kargs.get('channel_tree_addr', 0)
            self['first_attachment_addr'] = kargs.get(
                'first_attachment_addr',
                0,
            )
            self['first_event_addr'] = kargs.get('first_event_addr', 0)
            self['comment_addr'] = kargs.get('comment_addr', 0)
            self['abs_time'] = kargs.get('abs_time', int(time.time()) * 10 ** 9)
            self['tz_offset'] = kargs.get('tz_offset', 120)
            self['daylight_save_time'] = kargs.get('daylight_save_time', 60)
            self['time_flags'] = kargs.get('time_flags', 2)
            self['time_quality'] = kargs.get('time_quality', 0)
            self['flags'] = kargs.get('flags', 0)
            self['reserved4'] = kargs.get('reserved4', 0)
            self['start_angle'] = kargs.get('start_angle', 0)
            self['start_distance'] = kargs.get('start_distance', 0)

    def __bytes__(self):
        if PYVERSION_MAJOR >= 36:
            result = pack(v4c.FMT_HEADER_BLOCK, *self.values())
        else:
            result = pack(
                v4c.FMT_HEADER_BLOCK,
                *[self[key] for key in v4c.KEYS_HEADER_BLOCK]
            )
        return result


class HeaderList(dict):
    """HLBLOCK class"""

    __slots__ = ['address', ]

    def __init__(self, **kargs):
        super(HeaderList, self).__init__()

        try:
            self.address = address = kargs['address']
            stream = kargs['stream']
            stream.seek(address, SEEK_START)

            (self['id'],
             self['reserved0'],
             self['block_len'],
             self['links_nr'],
             self['first_dl_addr'],
             self['flags'],
             self['zip_type'],
             self['reserved1']) = unpack(
                v4c.FMT_HL_BLOCK,
                stream.read(v4c.HL_BLOCK_SIZE),
            )

            if self['id'] != b'##HL':
                message = 'Expected "##HL" block but found "{}"'
                raise MdfException(message.format(self['id']))

        except KeyError:

            self.address = 0
            self['id'] = b'##HL'
            self['reserved0'] = 0
            self['block_len'] = v4c.HL_BLOCK_SIZE
            self['links_nr'] = 1
            self['first_dl_addr'] = kargs.get('first_dl_addr', 0)
            self['flags'] = 1
            self['zip_type'] = kargs.get('zip_type', 0)
            self['reserved1'] = b'\x00' * 5

    def __bytes__(self):
        if PYVERSION_MAJOR >= 36:
            result = pack(v4c.FMT_HL_BLOCK, *self.values())
        else:
            result = pack(
                v4c.FMT_HL_BLOCK,
                *[self[key] for key in v4c.KEYS_HL_BLOCK]
            )
        return result


class SourceInformation(dict):
    """SIBLOCK class"""

    __slots__ = ['address', 'name', 'path', 'comment']

    def __init__(self, **kargs):
        super(SourceInformation, self).__init__()

        self.name = self.path = self.comment = ''

        if 'raw_bytes' in kargs:
            self.address = 0
            (self['id'],
             self['reserved0'],
             self['block_len'],
             self['links_nr'],
             self['name_addr'],
             self['path_addr'],
             self['comment_addr'],
             self['source_type'],
             self['bus_type'],
             self['flags'],
             self['reserved1']) = unpack(
                v4c.FMT_SOURCE_INFORMATION,
                kargs['raw_bytes'],
            )

            if self['id'] != b'##SI':
                message = 'Expected "##SI" block but found "{}"'
                raise MdfException(message.format(self['id']))

        elif 'stream' in kargs:
            self.address = address = kargs['address']
            stream = kargs['stream']
            stream.seek(address, SEEK_START)

            (self['id'],
             self['reserved0'],
             self['block_len'],
             self['links_nr'],
             self['name_addr'],
             self['path_addr'],
             self['comment_addr'],
             self['source_type'],
             self['bus_type'],
             self['flags'],
             self['reserved1']) = unpack(
                v4c.FMT_SOURCE_INFORMATION,
                stream.read(v4c.SI_BLOCK_SIZE),
            )

            if self['id'] != b'##SI':
                message = 'Expected "##SI" block but found "{}"'
                raise MdfException(message.format(self['id']))

        else:
            self.address = 0
            self['id'] = b'##SI'
            self['reserved0'] = 0
            self['block_len'] = v4c.SI_BLOCK_SIZE
            self['links_nr'] = 3
            self['name_addr'] = kargs.get('name_addr', 0)
            self['path_addr'] = kargs.get('path_addr', 0)
            self['comment_addr'] = kargs.get('comment_addr', 0)
            self['source_type'] = kargs.get('source_type', v4c.SOURCE_TOOL)
            self['bus_type'] = kargs.get('bus_type', v4c.BUS_TYPE_NONE)
            self['flags'] = 0
            self['reserved1'] = b'\x00' * 5

    def __bytes__(self):
        if PYVERSION_MAJOR >= 36:
            result = pack(v4c.FMT_SOURCE_INFORMATION, *self.values())
        else:
            result = pack(
                v4c.FMT_SOURCE_INFORMATION,
                *[self[key] for key in v4c.KEYS_SOURCE_INFORMATION]
            )
        return result


class SignalDataBlock(dict):
    """SDBLOCK class"""
    __slots__ = ['address', ]

    def __init__(self, **kargs):
        super(SignalDataBlock, self).__init__()

        try:
            self.address = address = kargs['address']
            stream = kargs['stream']
            stream.seek(address, SEEK_START)

            (self['id'],
             self['reserved0'],
             self['block_len'],
             self['links_nr']) = unpack(
                v4c.FMT_COMMON,
                stream.read(v4c.COMMON_SIZE),
            )
            self['data'] = stream.read(self['block_len'] - v4c.COMMON_SIZE)

            if self['id'] != b'##SD':
                message = 'Expected "##SD" block but found "{}"'
                raise MdfException(message.format(self['id']))

        except KeyError:

            self.address = 0
            self['id'] = b'##SD'
            self['reserved0'] = 0
            data = kargs['data']
            self['block_len'] = len(data) + v4c.COMMON_SIZE
            self['links_nr'] = 0
            self['data'] = data

    def __bytes__(self):
        fmt = v4c.FMT_DATA_BLOCK.format(self['block_len'] - v4c.COMMON_SIZE)
        keys = v4c.KEYS_DATA_BLOCK
        if PYVERSION_MAJOR >= 36:
            res = pack(fmt, *self.values())
        else:
            res = pack(fmt, *[self[key] for key in keys])
        return res


class TextBlock(dict):
    """common TXBLOCK and MDBLOCK class"""

    __slots__ = ['address', ]

    def __init__(self, **kargs):
        super(TextBlock, self).__init__()

        if 'stream' in kargs:
            stream = kargs['stream']
            self.address = address = kargs['address']

            stream.seek(address, SEEK_START)
            (self['id'],
             self['reserved0'],
             self['block_len'],
             self['links_nr']) = unpack(
                v4c.FMT_COMMON,
                stream.read(v4c.COMMON_SIZE),
            )

            size = self['block_len'] - v4c.COMMON_SIZE

            self['text'] = text = stream.read(size)

            if self['id'] not in (b'##TX', b'##MD'):
                message = 'Expected "##TX" or "##MD" block @{} but found "{}"'
                raise MdfException(message.format(hex(address), self['id']))

        else:

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

            text_length = size = len(text)

            self['id'] = b'##MD' if kargs.get('meta', False) else b'##TX'
            self['reserved0'] = 0
            self['block_len'] = text_length + v4c.COMMON_SIZE
            self['links_nr'] = 0
            self['text'] = text

        align = size % 8
        if align:
            self['block_len'] = size + v4c.COMMON_SIZE + 8 - align
        else:
            if text:
                if text[-1] not in (0, b'\0'):
                    self['block_len'] += 8
            else:
                self['block_len'] += 8

    def __bytes__(self):
        fmt = v4c.FMT_TEXT_BLOCK.format(self['block_len'] - v4c.COMMON_SIZE)
        if PYVERSION_MAJOR >= 36:
            result = pack(fmt, *self.values())
        else:
            result = pack(fmt, *[self[key] for key in v4c.KEYS_TEXT_BLOCK])
        return result

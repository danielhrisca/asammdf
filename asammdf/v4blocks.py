from __future__ import print_function, division
import sys
PYVERSION = sys.version_info[0]

import time
import warnings
import zlib

from hashlib import md5
from struct import unpack, pack, unpack_from
from functools import partial

try:
    from blosc import compress, decompress
    compress = partial(compress, clevel=7)

except ImportError:
    from zlib import compress, decompress

import numpy as np

from .v4constants import *


__all__ = ['AttachmentBlock',
           'Channel',
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
           'TextBlock']


class AttachmentBlock(dict):
    """ ATBLOCK class

    When adding new attachments only embedded attachemnts are allowed, with keyword argument *data* of type bytes"""
    def __init__(self, **kargs):
        super(AttachmentBlock, self).__init__()

        try:
            self.address = address = kargs['address']
            stream = kargs['file_stream']
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
             self['embedded_size']) = unpack(FMT_AT_COMMON, stream.read(AT_COMMON_SIZE))

            self['embedded_data'] = stream.read(self['embedded_size'])

        except KeyError:

            data = kargs['data']
            size = len(data)
            compression = kargs.get('compression', False)

            if compression:
                data = zlib.compress(data)
                original_size = size
                size = len(data)
                self['id'] = b'##AT'
                self['reserved0'] = 0
                self['block_len'] = AT_COMMON_SIZE + size
                self['links_nr'] = 4
                self['next_at_addr'] = 0
                self['file_name_addr'] = 0
                self['mime_addr'] = 0
                self['comment_addr'] = 0
                self['flags'] = FLAG_AT_EMBEDDED | FLAG_AT_MD5_VALID | FLAG_AT_COMPRESSED_EMBEDDED
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
                self['block_len'] = AT_COMMON_SIZE + size
                self['links_nr'] = 4
                self['next_at_addr'] = 0
                self['file_name_addr'] = 0
                self['mime_addr'] = 0
                self['comment_addr'] = 0
                self['flags'] = FLAG_AT_EMBEDDED | FLAG_AT_MD5_VALID
                self['creator_index'] = 0
                self['reserved1'] = 0
                md5_worker = md5()
                md5_worker.update(data)
                self['md5_sum'] = md5_worker.digest()
                self['original_size'] = size
                self['embedded_size'] = size
                self['embedded_data'] = data

    def extract(self):
        if self['flags'] & FLAG_AT_EMBEDDED:
            if self['flags'] & FLAG_AT_COMPRESSED_EMBEDDED:
                data = zlib.decompress(self['embedded_data'])
            else:
                data = self['embedded_data']
            if self['flags'] & FLAG_AT_MD5_VALID:
                md5_worker = md5()
                md5_worker.update(data)
                md5_sum = md5_worker.digest()
                if self['md5_sum'] == md5_sum:
                    return data
                else:
                    warnings.warn('ATBLOCK md5sum="{}" and embedded data md5sum="{}"'.format(self['md5_sum'], md5_sum))
        else:
            warnings.warn('extarnal attachments not supported')

    def __bytes__(self):
        fmt = FMT_AT_COMMON + '{}s'.format(self['embedded_size'])
        return pack(fmt, *[self[key] for key in KEYS_AT_BLOCK])

class Channel(dict):
    """ CNBLOCK class"""
    def __init__(self, **kargs):
        super(Channel, self).__init__()

        self.name = ''

        try:
            self.address = address = kargs['address']
            stream = kargs['file_stream']
            stream.seek(address, SEEK_START)

            (self['id'],
             self['reserved0'],
             self['block_len'],
             self['links_nr'],
             self['next_ch_addr'],
             self['component_addr'],
             self['name_addr'],
             self['source_addr'],
             self['conversion_addr'],
             self['data_block_addr'],
             self['unit_addr'],
             self['comment_addr'],
             self['channel_type'],
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
             self['upper_ext_limit']) = unpack(FMT_CHANNEL, stream.read(CN_BLOCK_SIZE))

        except KeyError:

            self.address = 0

            self['id'] = b'##CN'
            self['reserved0'] = 0
            self['block_len'] = CN_BLOCK_SIZE
            self['links_nr'] = 8
            self['next_ch_addr'] = 0
            self['component_addr'] = 0
            self['name_addr'] = 0
            self['source_addr'] = 0
            self['conversion_addr'] = 0
            self['data_block_addr'] = 0
            self['unit_addr'] = 0
            self['comment_addr'] = 0
            self['channel_type'] = kargs.get('channel_type', 0)
            self['sync_type'] = kargs.get('sync_type', 0)
            self['data_type'] = kargs.get('data_type', 0)
            self['bit_offset'] = kargs.get('bit_offset', 0)
            self['byte_offset'] = kargs.get('byte_offset', 8)
            self['bit_count'] = kargs.get('bit_count', 8)
            self['flags'] = 28
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

    def __bytes__(self):
        return pack(FMT_CHANNEL, *[self[key] for key in KEYS_CHANNEL])


class ChannelGroup(dict):
    """CGBLOCK class"""
    def __init__(self, **kargs):
        super(ChannelGroup, self).__init__()

        try:
            self.address = address = kargs['address']
            stream = kargs['file_stream']
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
             self['invalidation_bytes_nr']) = unpack(FMT_CHANNEL_GROUP, stream.read(CG_BLOCK_SIZE))

        except KeyError:
            self.address = 0
            self['id'] = kargs.get('id', '##CG'.encode('utf-8'))
            self['reserved0'] = kargs.get('reserved0', 0)
            self['block_len'] = kargs.get('block_len', CG_BLOCK_SIZE)
            self['links_nr'] = kargs.get('links_nr', 6)
            self['next_cg_addr'] = kargs.get('next_cg_addr', 0)
            self['first_ch_addr'] = kargs.get('first_ch_addr', 0)
            self['acq_name_addr'] = kargs.get('acq_name_addr', 0)
            self['acq_source_addr'] = kargs.get('acq_source_addr', 0)
            self['first_sample_reduction_addr'] = kargs.get('first_sample_reduction_addr', 0)
            self['comment_addr'] = kargs.get('comment_addr', 0)
            self['record_id'] = kargs.get('record_id', 1)
            self['cycles_nr'] = kargs.get('cycles_nr', 0)
            self['flags'] = kargs.get('flags', 0)
            self['path_separator'] = kargs.get('path_separator', 0)
            self['reserved1'] = kargs.get('reserved1', 0)
            self['samples_byte_nr'] = kargs.get('samples_byte_nr', 0)
            self['invalidation_bytes_nr'] = kargs.get('invalidation_bytes_nr', 0)

    def __bytes__(self):
        return pack(FMT_CHANNEL_GROUP, *[self[key] for key in KEYS_CHANNEL_GROUP])


class ChannelConversion(dict):
    """CCBLOCK class"""
    def __init__(self, **kargs):
        super(ChannelConversion, self).__init__()

        try:
            self.address = address = kargs['address']
            stream = kargs['file_stream']
            stream.seek(address, SEEK_START)

            (self['id'],
             self['reserved0'],
             self['block_len'],
             self['links_nr']) = unpack(FMT_COMMON, stream.read(COMMON_SIZE))

            block = stream.read(self['block_len'] - COMMON_SIZE)

            conv = unpack_from('B', block, self['links_nr'] * 8)[0]

            if conv == CONVERSION_TYPE_NON:
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
                 self['max_phy_value']) = unpack('<4Q2B3H2d', block)
            elif conv == CONVERSION_TYPE_LIN:
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
                 self['a']) = unpack('<4Q2B3H4d', block)
            elif conv == CONVERSION_TYPE_RAT:
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
                 self['P6']) = unpack('<4Q2B3H8d', block)

            elif conv == CONVERSION_TYPE_ALG:
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
                 self['max_phy_value']) = unpack('<5Q2B3H2d', block)

            elif conv in (CONVERSION_TYPE_TABI, CONVERSION_TYPE_TAB):
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
                 self['max_phy_value']) = unpack_from('<4Q2B3H2d', block)

                nr = self['val_param_nr']
                values = unpack('<{}d'.format(nr), block[56:])
                for i in range(nr // 2):
                    (self['raw_{}'.format(i)],
                     self['phys_{}'.format(i)]) = values[i*2], values[2*i+1]

            elif conv == CONVERSION_TYPE_RTAB:
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
                 self['max_phy_value']) = unpack_from('<4Q2B3H2d', block)
                nr = self['val_param_nr']
                values = unpack('<{}d'.format(nr), block[56:])
                for i in range((nr - 1) // 3):
                    (self['lower_{}'.format(i)],
                     self['upper_{}'.format(i)],
                     self['phys_{}'.format(i)]) = values[i*3], values[3*i+1], values[3*i+2]
                self['default'] = unpack('<d', block[-8:])[0]

            elif conv == CONVERSION_TYPE_TABX:
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
                 self['max_phy_value']) = unpack_from('<2B3H2d', block, 32 + links_nr * 8)

                values = unpack_from('<{}d'.format(links_nr - 1), block, 32 + links_nr * 8 + 24)
                for i, val in enumerate(values):
                    self['val_{}'.format(i)] = val

            elif conv == CONVERSION_TYPE_RTABX:
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
                 self['max_phy_value']) = unpack_from('<2B3H2d', block, 32 + links_nr * 8)

                values = unpack_from('<{}d'.format( (links_nr - 1) * 2 ), block, 32 + links_nr * 8 + 24)
                for i in range(self['val_param_nr'] // 2):
                    j = 2 * i
                    self['lower_{}'.format(i)] = values[j]
                    self['upper_{}'.format(i)] = values[j+1]

            elif conv == CONVERSION_TYPE_TTAB:
                (self['name_addr'],
                 self['unit_addr'],
                 self['comment_addr'],
                 self['inv_conv_addr']) = unpack_from('<4Q', block)

                links_nr = self['links_nr'] - 4

                links = unpack_from('<{}Q'.format(links_nr), block, 32)
                for i, link in enumerate(links[:-1]):
                    self['text_{}'.format(i)] = link

                (self['conversion_type'],
                 self['precision'],
                 self['flags'],
                 self['ref_param_nr'],
                 self['val_param_nr'],
                 self['min_phy_value'],
                 self['max_phy_value']) = unpack_from('<2B3H2d', block, 32 + links_nr * 8)

                values = unpack_from('<{}d'.format(self['val_param_nr']), block, 32 + links_nr * 8 + 24)
                for i, val in enumerate(values[:-1]):
                    self['val_{}'.format(i)] = val
                self['val_default'] = values[-1]

            elif conv == CONVERSION_TYPE_TRANS:
                (self['name_addr'],
                 self['unit_addr'],
                 self['comment_addr'],
                 self['inv_conv_addr']) = unpack_from('<4Q', block)

                links_nr = self['links_nr'] - 4

                links = unpack_from('<{}Q'.format(links_nr), block, 32)

                for i in range((links_nr -1) // 2):
                    j = 2 * i
                    self['input_{}_addr'.format(i)] = links[j]
                    self['output_{}_addr'.format(i)] = links[j+1]
                self['default_addr'] = links[-1]

                (self['conversion_type'],
                 self['precision'],
                 self['flags'],
                 self['ref_param_nr'],
                 self['val_param_nr'],
                 self['min_phy_value'],
                 self['max_phy_value']) = unpack_from('<2B3H2d', block, 32 + links_nr * 8)

        except KeyError:

            self.address = 0
            self['id'] = '##CC'.encode('utf-8')
            self['reserved0'] = 0

            if kargs['conversion_type'] == CONVERSION_TYPE_NON:
                self['block_len'] = CC_NONE_BLOCK_SIZE
                self['links_nr'] = 4
                self['name_addr'] = 0
                self['unit_addr'] = 0
                self['comment_addr'] = 0
                self['inv_conv_addr'] = 0
                self['conversion_type'] = CONVERSION_TYPE_NON
                self['precision'] = 1
                self['flags'] = 0
                self['ref_param_nr'] = 0
                self['val_param_nr'] = 0
                self['min_phy_value'] = kargs.get('min_phy_value', 0)
                self['max_phy_value'] = kargs.get('max_phy_value', 0)
            elif kargs['conversion_type'] == CONVERSION_TYPE_LIN:
                self['block_len'] = kargs.get('block_len', CC_LIN_BLOCK_SIZE)
                self['links_nr'] = kargs.get('links_nr', 4)
                self['name_addr'] = kargs.get('name_addr', 0)
                self['unit_addr'] = kargs.get('unit_addr', 0)
                self['comment_addr'] = kargs.get('comment_addr', 0)
                self['inv_conv_addr'] = kargs.get('inv_conv_addr', 0)
                self['conversion_type'] = CONVERSION_TYPE_LIN
                self['precision'] = kargs.get('precision', 1)
                self['flags'] = kargs.get('flags', 0)
                self['ref_param_nr'] = kargs.get('ref_param_nr', 0)
                self['val_param_nr'] = kargs.get('val_param_nr', 2)
                self['min_phy_value'] = kargs.get('min_phy_value', 0)
                self['max_phy_value'] = kargs.get('max_phy_value', 0)
                self['b'] = kargs.get('b', 0)
                self['a'] = kargs.get('a', 1)
            elif kargs['conversion_type'] == CONVERSION_TYPE_ALG:
                self['block_len'] = kargs.get('block_len', CC_ALG_BLOCK_SIZE)
                self['links_nr'] = kargs.get('links_nr', 5)
                self['name_addr'] = kargs.get('name_addr', 0)
                self['unit_addr'] = kargs.get('unit_addr', 0)
                self['comment_addr'] = kargs.get('comment_addr', 0)
                self['inv_conv_addr'] = kargs.get('inv_conv_addr', 0)
                self['conv_text_addr'] = kargs.get('conv_text_addr', 0)
                self['conversion_type'] = CONVERSION_TYPE_ALG
                self['precision'] = kargs.get('precision', 1)
                self['flags'] = kargs.get('flags', 0)
                self['ref_param_nr'] = kargs.get('ref_param_nr', 1)
                self['val_param_nr'] = kargs.get('val_param_nr', 0)
                self['min_phy_value'] = kargs.get('min_phy_value', 0)
                self['max_phy_value'] = kargs.get('max_phy_value', 0)
            elif kargs['conversion_type'] == CONVERSION_TYPE_TABX:

                self['block_len'] = ((kargs['links_nr'] - 5) * 8 * 2) + 88
                self['links_nr'] = kargs['links_nr']
                self['name_addr'] = kargs.get('name_addr', 0)
                self['unit_addr'] = kargs.get('unit_addr', 0)
                self['comment_addr'] = kargs.get('comment_addr', 0)
                self['inv_conv_addr'] = kargs.get('inv_conv_addr', 0)
                for i in range(kargs['links_nr'] - 5):
                    self['text_{}'.format(i)] = 0
                self['default_addr'] = kargs.get('default_addr', 0)
                self['conversion_type'] = CONVERSION_TYPE_TABX
                self['precision'] = kargs.get('precision', 0)
                self['flags'] = kargs.get('flags', 0)
                self['ref_param_nr'] = kargs.get('ref_param_nr', kargs['links_nr'] - 4)
                self['val_param_nr'] = kargs.get('val_param_nr', kargs['links_nr'] - 5)
                self['min_phy_value'] = kargs.get('min_phy_value', 0)
                self['max_phy_value'] = kargs.get('max_phy_value', 0)
                for i in range(kargs['links_nr'] - 5):
                    self['val_{}'.format(i)] = kargs['val_{}'.format(i)]
            elif kargs['conversion_type'] == CONVERSION_TYPE_RTABX:
                self['block_len'] = ((kargs['links_nr'] - 5) * 8 * 3) + 88
                self['links_nr'] = kargs['links_nr']
                self['name_addr'] = kargs.get('name_addr', 0)
                self['unit_addr'] = kargs.get('unit_addr', 0)
                self['comment_addr'] = kargs.get('comment_addr', 0)
                self['inv_conv_addr'] = kargs.get('inv_conv_addr', 0)
                for i in range(kargs['links_nr'] - 5):
                    self['text_{}'.format(i)] = 0
                self['default_addr'] = kargs.get('default_addr', 0)
                self['conversion_type'] = CONVERSION_TYPE_RTABX
                self['precision'] = kargs.get('precision', 0)
                self['flags'] = kargs.get('flags', 0)
                self['ref_param_nr'] = kargs['links_nr'] - 4
                self['val_param_nr'] = (kargs['links_nr'] - 5) * 2
                self['min_phy_value'] = kargs.get('min_phy_value', 0)
                self['max_phy_value'] = kargs.get('max_phy_value', 0)
                for i in range(kargs['links_nr'] - 5):
                    self['lower_{}'.format(i)] = kargs['lower_{}'.format(i)]
                    self['upper_{}'.format(i)] = kargs['upper_{}'.format(i)]

    def __bytes__(self):
        fmt = '<4sI{}Q2B3H{}d'.format(self['links_nr'] + 2, self['val_param_nr'] + 2)
        if self['conversion_type'] == CONVERSION_TYPE_NON:
            keys = KEYS_CONVERSION_NONE
        elif self['conversion_type'] == CONVERSION_TYPE_LIN:
            keys = KEYS_CONVERSION_LINEAR
        elif self['conversion_type'] == CONVERSION_TYPE_RAT:
            keys = KEYS_CONVERSION_RAT
        elif self['conversion_type'] == CONVERSION_TYPE_ALG:
            keys = KEYS_CONVERSION_ALGEBRAIC
        elif self['conversion_type'] in (CONVERSION_TYPE_TABI, CONVERSION_TYPE_TAB):
            keys = KEYS_CONVERSION_NONE
            for i in range(self['val_param_nr'] // 2):
                keys += ('raw_{}'.format(i), 'phys_{}'.format(i))
        elif self['conversion_type'] == CONVERSION_TYPE_RTAB:
            keys = KEYS_CONVERSION_NONE
            for i in range(self['val_param_nr']):
                keys += ('lower{}'.format(i), 'upper_{}'.format(i), 'phys_{}'.format(i))
            keys += ('default',)
        elif self['conversion_type'] == CONVERSION_TYPE_TABX:
            keys = ('id',
                    'reserved0',
                    'block_len',
                    'links_nr',
                    'name_addr',
                    'unit_addr',
                    'comment_addr',
                    'inv_conv_addr')
            keys += tuple('text_{}'.format(i) for i in range(self['links_nr'] - 4 - 1))
            keys += ('default_addr',)
            keys += ('conversion_type',
                     'precision',
                     'flags',
                     'ref_param_nr',
                     'val_param_nr',
                     'min_phy_value',
                     'max_phy_value')
            keys += tuple('val_{}'.format(i) for i in range(self['val_param_nr']))
        elif self['conversion_type'] == CONVERSION_TYPE_RTABX:
            keys = ('id',
                    'reserved0',
                    'block_len',
                    'links_nr',
                    'name_addr',
                    'unit_addr',
                    'comment_addr',
                    'inv_conv_addr')
            keys += tuple('text_{}'.format(i) for i in range(self['links_nr'] - 4 - 1))
            keys += ('default_addr',)
            keys += ('conversion_type',
                     'precision',
                     'flags',
                     'ref_param_nr',
                     'val_param_nr',
                     'min_phy_value',
                     'max_phy_value')
            for i in range(self['val_param_nr'] // 2):
                keys += ('lower_{}'.format(i), 'upper_{}'.format(i))
        elif self['conversion_type'] == CONVERSION_TYPE_TTAB:
            keys = ('id',
                    'reserved0',
                    'block_len',
                    'links_nr',
                    'name_addr',
                    'unit_addr',
                    'comment_addr',
                    'inv_conv_addr')
            keys += tuple('text_{}'.format(i) for i in range(self['links_nr'] - 4))
            keys += ('conversion_type',
                     'precision',
                     'flags',
                     'ref_param_nr',
                     'val_param_nr',
                     'min_phy_value',
                     'max_phy_value')
            keys += tuple('val_{}'.format(i) for i in range(self['val_param_nr'] -1))
            keys += ('val_default',)
        elif self['conversion_type'] == CONVERSION_TYPE_TRANS:
            keys = ('id',
                    'reserved0',
                    'block_len',
                    'links_nr',
                    'name_addr',
                    'unit_addr',
                    'comment_addr',
                    'inv_conv_addr')
            for i in range((self['links_nr'] - 4 -1) // 2):
                keys += ('input_{}_addr'.format(i), 'output_{}_addr'.format(i))
            keys += ('default_addr',
                     'conversion_type',
                     'precision',
                     'flags',
                     'ref_param_nr',
                     'val_param_nr',
                     'min_phy_value',
                     'max_phy_value')
            keys += tuple('val_{}'.format(i) for i in range(self['val_param_nr'] -1))
            keys += ('val_default',)

        return pack(fmt, *[self[key] for key in keys])


class DataBlock(dict):
    """DTBLOCK class
    Raw channel dta can be compressed to save RAM; set the *compression* keyword argument to True when instantiating the object

    Parameters
    ----------
    compression : bool
        enable raw channel data compression in RAM
    address : int
        DTBLOCK address inside the file
    file_stream : int
        file handle

    """
    def __init__(self, **kargs):
        super(DataBlock, self).__init__()

        self.compression = kargs.get('compression', False)

        try:
            self.address = address = kargs['address']
            stream = kargs['file_stream']
            stream.seek(address, SEEK_START)

            (self['id'],
             self['reserved0'],
             self['block_len'],
             self['links_nr']) = unpack(FMT_COMMON, stream.read(COMMON_SIZE))
            self['data'] = stream.read(self['block_len'] - COMMON_SIZE)

        except KeyError:

            self['id'] = b'##DT'
            self['reserved0'] = 0
            self['block_len'] = len(kargs['data']) + COMMON_SIZE
            self['links_nr'] = 0
            self['data'] = kargs['data']

    def __setitem__(self, item, value):
        if item == 'data':
            if self.compression:
                super(DataBlock, self).__setitem__(item, compress(value))
            else:
                super(DataBlock, self).__setitem__(item, value)
        else:
            super(DataBlock, self).__setitem__(item, value)

    def __getitem__(self, item):
        if item == 'data' and self.compression:
            return decompress(super(DataBlock, self).__getitem__(item))
        else:
            return super(DataBlock, self).__getitem__(item)

    def __bytes__(self):
        return pack(FMT_DATA_BLOCK.format(self['block_len'] - COMMON_SIZE), *[self[key] for key in KEYS_DATA_BLOCK])



class DataZippedBlock(dict):
    """DZBLOCK class
    Raw channel dta can be compressed to save RAM; set the *compression* keyword argument to True when instantiating the object

    Parameters
    ----------
    compression : bool
        enable raw channel data compression in RAM
    address : int
        DTBLOCK address inside the file
    file_stream : int
        file handle

    """
    def __init__(self, **kargs):
        super(DataZippedBlock, self).__init__()

        self.prevent_data_setitem = True
        try:
            self.address = address = kargs['address']
            stream = kargs['file_stream']
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
             self['zip_size'],) = unpack(FMT_DZ_COMMON, stream.read(DZ_COMMON_SIZE))

            self['data'] = stream.read(self['zip_size'])

        except KeyError:
            self.prevent_data_setitem = False

            data = kargs['data']

            self['id'] = b'##DZ'
            self['reserved0'] = 0

            self['links_nr'] = 0
            self['original_type'] = kargs.get('original_type', 'DT')
            self['zip_type'] = karg.get('zip_type', FLAG_DZ_DEFLATE)
            self['reserved1'] = 0
            self['param'] = 0 if self['zip_type'] == FLAG_DZ_DEFLATE else kargs['param']

            # since prevent_data_setitem is False the rest of the keys will be handled by __setitem__
            self['data'] = data

        self.prevent_data_setitem = False
        self.return_unzipped = True

    def __setitem__(self, item, value):
        if item == 'data' and self.prevent_data_setitem == False:
            data = value
            self['original_size'] = len(data)

            if self['zip_type'] == FLAG_DZ_DEFLATE:
                data = zlib.compress(data)
            else:
                cols = self['param']
                lines = self['original_size'] // cols

                nd = np.fromstring(data[:lines*cols], dtype=np.uint8).reshape((lines, cols))
                data = nd.transpose().tostring() + data[lines*cols:]

                data = zlib.compress(data)

            self['zip_size'] = len(data)
            self['block_len'] = self['zip_size'] + DZ_COMMON_SIZE
            super(DataZippedBlock, self).__setitem__(item, data)
        else:
            super(DataZippedBlock, self).__setitem__(item, value)

    def __getitem__(self, item):
        if item == 'data':
            if self.return_unzipped:
                data = super(DataZippedBlock, self).__getitem__(item)
                data = zlib.decompress(data)
                if self['zip_type'] == FLAG_DZ_DEFLATE:
                    return data
                else:
                    cols = self['param']
                    lines = self['original_size'] // cols

                    nd = np.fromstring(data[:lines*cols], dtype=np.uint8).reshape((cols, lines))
                    data = nd.transpose().tostring() + data[lines*cols:]

                    return data
            else:
                return super(DataZippedBlock, self).__getitem__(item)
        else:
            return super(DataZippedBlock, self).__getitem__(item)

    def __bytes__(self):
        fmt = FMT_DZ_COMMON + '{}s'.format(self['zip_size'])
        self.return_unzipped = False
        data = pack(fmt, *[self[key] for key in KEYS_DZ_BLOCK])
        self.return_unzipped = True
        return data


class DataGroup(dict):
    """DGBLOCK class"""
    def __init__(self, **kargs):
        super(DataGroup, self).__init__()

        try:
            self.address = address = kargs['address']
            stream = kargs['file_stream']
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
             self['reserved1']) = unpack(FMT_DATA_GROUP, stream.read(DG_BLOCK_SIZE))

        except KeyError:

            self.address = 0
            self['id'] = kargs.get('id', '##DG'.encode('utf-8'))
            self['reserved0'] = kargs.get('reserved0', 0)
            self['block_len'] = kargs.get('block_len', DG_BLOCK_SIZE)
            self['links_nr'] = kargs.get('links_nr', 4)
            self['next_dg_addr'] = kargs.get('next_dg_addr', 0)
            self['first_cg_addr'] = kargs.get('first_cg_addr', 0)
            self['data_block_addr'] = kargs.get('data_block_addr', 0)
            self['comment_addr'] = kargs.get('comment_addr', 0)
            self['record_id_len'] = kargs.get('record_id_len', 0)
            self['reserved1'] = kargs.get('reserved1', b'\00'*7)

    def __bytes__(self):
        return pack(FMT_DATA_GROUP, *[self[key] for key in KEYS_DATA_GROUP])


class DataList(dict):
    """DLBLOCK class"""
    def __init__(self, **kargs):
        super(DataList, self).__init__()

        try:
            self.address = address = kargs['address']
            stream = kargs['file_stream']
            stream.seek(address, SEEK_START)

            (self['id'],
             self['reserved0'],
             self['block_len'],
             self['links_nr']) = unpack(FMT_COMMON, stream.read(COMMON_SIZE))

            self['next_dl_addr'] = unpack('<Q', stream.read(8))[0]

            links = unpack('<{}Q'.format(self['links_nr'] - 1), stream.read( (self['links_nr'] - 1) * 8 ))

            for i, addr in enumerate(links):
                self['data_block_addr{}'.format(i)] = addr

            (self['flags'],
             self['reserved1'],
             self['data_block_nr'],
             self['data_block_len']) = unpack('<B3sIQ', stream.read(16))

        except KeyError:

            self.address = 0
            self['id'] = kargs.get('id', '##DL'.encode('utf-8'))
            self['reserved0'] = kargs.get('reserved0', 0)
            self['block_len'] = kargs.get('block_len', CN_BLOCK_SIZE)
            self['links_nr'] = kargs.get('links_nr', 8)
            self['next_dl_addr'] = kargs.get('next_dl_addr', 0)

            for i in range(self['links_nr'] - 1):
                self['data_block_addr{}'.format(i)] = kargs.get('data_block_addr{}'.format(i), 0)

            self['flags'] = kargs.get('flags', 1)
            self['reserved1'] = kargs.get('reserved1', b'\00'*3)
            self['data_block_nr'] = kargs.get('data_block_nr', 1)
            self['data_block_len'] = kargs.get('data_block_len', 1)

    def __bytes__(self):
        keys = ('id', 'reserved0', 'block_len', 'links_nr', 'next_dl_addr')
        keys += tuple('data_block_addr{}'.format(i) for i in range(self['links_nr'] - 1))
        keys += ('flags', 'reserved1', 'data_block_nr', 'data_block_len')
        return pack(FMT_DATA_LIST.format(self['links_nr']), *[self[key] for key in keys])


class FileIdentificationBlock(dict):
    """IDBLOCK class"""
    def __init__(self, **kargs):

        super(FileIdentificationBlock, self).__init__()

        self.address = 0

        try:

            stream = kargs['file_stream']
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
             self['unfinalized_custom_flags']) = unpack(FMT_IDENTIFICATION_BLOCK, stream.read(IDENTIFICATION_BLOCK_SIZE))

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
        return pack(FMT_IDENTIFICATION_BLOCK, *[self[key] for key in KEYS_IDENTIFICATION_BLOCK])


class FileHistory(dict):
    """FHBLOCK class"""
    def __init__(self, **kargs):
        super(FileHistory, self).__init__()

        try:
            self.address = address = kargs['address']
            stream = kargs['file_stream']
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
             self['reserved1']) = unpack(FMT_FILE_HISTORY, stream.read(FH_BLOCK_SIZE))

        except KeyError:
            self['id'] = kargs.get('id', '##FH'.encode('utf-8'))
            self['reserved0'] = kargs.get('reserved0', 0)
            self['block_len'] = kargs.get('block_len', FH_BLOCK_SIZE)
            self['links_nr'] = kargs.get('links_nr', 2)
            self['next_fh_addr'] = kargs.get('next_fh_addr', 0)
            self['comment_addr'] = kargs.get('comment_addr', 0)
            self['abs_time'] = kargs.get('abs_time', int(time.time()) * 10**9)
            self['tz_offset'] = kargs.get('tz_offset', 120)
            self['daylight_save_time'] = kargs.get('daylight_save_time', 60)
            self['time_flags'] = kargs.get('time_flags', 2)
            self['reserved1'] = kargs.get('reserved1', b'\x00'*3)

    def __bytes__(self):
        return pack(FMT_FILE_HISTORY, *[self[key] for key in KEYS_FILE_HISTORY])


class HeaderBlock(dict):
    """HDBLOCK class"""
    def __init__(self, **kargs):
        super(HeaderBlock, self).__init__()

        try:
            self.address = address = kargs['address']
            stream = kargs['file_stream']
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
             self['start_distance']) = unpack(FMT_HEADER_BLOCK, stream.read(HEADER_BLOCK_SIZE))

        except KeyError:

            self['id'] = '##HD'.encode('utf-8')
            self['reserved3'] = kargs.get('reserved3' , 0)
            self['block_len'] = kargs.get('block_len' , HEADER_BLOCK_SIZE)
            self['links_nr'] = kargs.get('links_nr' , 6)
            self['first_dg_addr'] = kargs.get('first_dg_addr' , 0)
            self['file_history_addr'] = kargs.get('file_history_addr' , 0)
            self['channel_tree_addr'] = kargs.get('channel_tree_addr' , 0)
            self['first_attachment_addr'] = kargs.get('first_attachment_addr' , 0)
            self['first_event_addr'] = kargs.get('first_event_addr' , 0)
            self['comment_addr'] = kargs.get('comment_addr' , 0)
            self['abs_time'] = kargs.get('abs_time' , int(time.time()) * 10**9)
            self['tz_offset'] = kargs.get('tz_offset' , 120)
            self['daylight_save_time'] = kargs.get('daylight_save_time' , 60)
            self['time_flags'] = kargs.get('time_flags' , 2)   #offset valid
            self['time_quality'] = kargs.get('time_quality' , 0) #time source PC
            self['flags'] = kargs.get('flags' , 0)
            self['reserved4'] = kargs.get('reserved4', 0)
            self['start_angle'] = kargs.get('start_angle' , 0)
            self['start_distance'] = kargs.get('start_distance' , 0)

    def __bytes__(self):
        return pack(FMT_HEADER_BLOCK, *[self[key] for key in KEYS_HEADER_BLOCK])


class HeaderList(dict):
    """HLBLOCK class"""
    def __init__(self, **kargs):
        super(HeaderList, self).__init__()

        try:
            self.address = address = kargs['address']
            stream = kargs['file_stream']
            stream.seek(address, SEEK_START)

            (self['id'],
             self['reserved0'],
             self['block_len'],
             self['links_nr'],
             self['first_dl_addr'],
             self['flags'],
             self['zip_type'],
             self['reserved1']) = unpack(FMT_HL_BLOCK, stream.read(HL_BLOCK_SIZE))

        except KeyError:

            self.address = 0
            self['id'] = b'##HL'
            self['reserved0'] = 0
            self['block_len'] = HL_BLOCK_SIZE
            self['links_nr'] = 1
            self['first_dl_addr'] = 0
            self['flags'] = 1
            self['zip_type'] = 0
            self['reserved1'] = b'\x00' * 5

    def __bytes__(self):
        return pack(FMT_HL_BLOCK, *[self[key] for key in KEYS_HL_BLOCK])


class SourceInformation(dict):
    """SIBLOCK class"""
    def __init__(self, **kargs):
        super(SourceInformation, self).__init__()

        try:
            self.address = address = kargs['address']
            stream = kargs['file_stream']
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
             self['reserved1']) = unpack(FMT_SOURCE_INFORMATION, stream.read(SI_BLOCK_SIZE))

        except KeyError:
            self.address = 0
            self['id'] = b'##SI'
            self['reserved0'] = 0
            self['block_len'] = SI_BLOCK_SIZE
            self['links_nr'] = 3
            self['name_addr'] = 0
            self['path_addr'] = 0
            self['comment_addr'] = 0
            self['source_type'] = kargs.get('source_type', SOURCE_TOOL)
            self['bus_type'] = kargs.get('bus_type', BUS_TYPE_NONE)
            self['flags'] = 0
            self['reserved1'] = b'\x00' * 5

    def __bytes__(self):
        return pack(FMT_SOURCE_INFORMATION, *[self[key] for key in KEYS_SOURCE_INFORMATION])


class SignalDataBlock(dict):
    """SDBLOCK class"""
    def __init__(self, **kargs):
        super(SignalDataBlock, self).__init__()

        try:
            self.address = address = kargs['address']
            stream = kargs['file_stream']
            stream.seek(address, SEEK_START)

            (self['id'],
             self['reserved0'],
             self['block_len'],
             self['links_nr']) = unpack(FMT_COMMON, stream.read(COMMON_SIZE))
            self['data'] = stream.read(self['block_len'] - COMMON_SIZE)

        except KeyError:
            self.address = 0
            self['id'] = b'##SD'
            self['reserved0'] = 0
            data = kargs['data']
            self['block_len'] = len(data) + COMMON_SIZE
            self['links_nr'] = 0
            self['data'] = data

    def __bytes__(self):
        fmt = FMT_DATA_BLOCK.format(self['block_len'] - COMMON_SIZE)
        keys = KEYS_DATA_BLOCK
        res = pack(fmt, *[self[key] for key in keys])
        # 8 byte alignment
        size = len(res)
        if size % 8:
            res += b'\x00' * (8 - size%8)
        return res


class TextBlock(dict):
    """common TXBLOCK and MDBLOCK class"""
    def __init__(self, **kargs):
        super(TextBlock, self).__init__()

        try:
            stream = kargs['file_stream']
            self.address = address = kargs['address']

            stream.seek(address, SEEK_START)
            (self['id'],
             self['reserved0'],
             self['block_len'],
             self['links_nr']) = unpack(FMT_COMMON, stream.read(COMMON_SIZE))

            size = self['block_len'] - COMMON_SIZE

            self['text'] = text = stream.read(size)

            self.text_str = text.decode('utf-8').strip('\x00')

        except KeyError:

            self.address = 0
            text = kargs['text']
            if isinstance(text, str):
                self.text_str = text
                text = text.encode('utf-8')
            elif isinstance(text, bytes):
                self.text_str = text.decode('utf-8')
            elif isinstance(text, unicode):
                self.text_str = text
                text = text.encode('utf-8')
            text_length = len(text)
            align = text_length % 8
            if align == 0 and text[-1] == b'\x00':
                padding = 0
            else:
                padding = 8 - align

            self['id'] = kargs['id']
            self['reserved0'] = 0
            self['block_len'] = text_length + padding + COMMON_SIZE
            self['links_nr'] = 0
            self['text'] = text + b'\00' * padding


    @classmethod
    def from_text(cls, text, meta=False):
        """Create a TextBlock from a str or bytes

        Parameters
        ----------
        text : str | bytes
            input text
        meta : bool
            enable meta text block

        Returns
        -------

        Examples
        --------
        >>> t = TextBlock.from_text(b'speed')
        >>> t['id']
        b'##TX'
        >>> t.text_str
        speed
        >>> t = TextBlock.from_text('mass', meta=True)
        >>> t['id']
        b'##MD'

        """
        kargs = {}

        kargs['id'] = b'##MD' if meta else b'##TX'
        kargs['text'] = text

        return cls(**kargs)

    def __bytes__(self):
        return pack(FMT_TEXT_BLOCK.format(self['block_len'] - COMMON_SIZE), *[self[key] for key in KEYS_TEXT_BLOCK])

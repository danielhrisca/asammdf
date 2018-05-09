"""
ASAM MDF version 4 file format module
"""

from __future__ import division, print_function

import logging
import xml.etree.ElementTree as ET
import os
import sys
from collections import defaultdict
from copy import deepcopy
from functools import reduce
from hashlib import md5
from itertools import chain
from math import ceil
from struct import unpack, unpack_from
from tempfile import TemporaryFile
from zlib import decompress

from numpy import (
    arange,
    argwhere,
    array,
    array_equal,
    concatenate,
    dtype,
    flip,
    float64,
    frombuffer,
    interp,
    ones,
    packbits,
    roll,
    transpose,
    uint8,
    uint16,
    uint64,
    union1d,
    unpackbits,
    zeros,
    uint32,
)
from numpy.core.defchararray import encode, decode
from numpy.core.records import fromarrays, fromstring
from canmatrix.formats import loads

from . import v4_constants as v4c
from .signal import Signal
from .conversion_utils import conversion_transfer
from .utils import (
    CHANNEL_COUNT,
    CONVERT_LOW,
    CONVERT_MINIMUM,
    MdfException,
    SignalSource,
    as_non_byte_sized_signed_int,
    fix_dtype_fields,
    fmt_to_datatype_v4,
    get_fmt_v4,
    get_min_max,
    get_unique_name,
    get_text_v4,
    debug_channel,
    extract_cncomment_xml,
    validate_memory_argument,
    validate_version_argument,
    count_channel_groups,
    info_to_datatype_v4,
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
    EventBlock,
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

PYVERSION = sys.version_info[0]
if PYVERSION == 2:
    # pylint: disable=W0622
    from .utils import bytes
    # pylint: enable=W0622

logger = logging.getLogger('asammdf')

__all__ = ['MDF4', ]


def write_cc(conversion, defined_texts, blocks=None, address=None, stream=None):
    if conversion:
        if stream:
            tell = stream.tell
            write = stream.write
            stream.seek(0, 2)
        if conversion.name:
            tx_block = TextBlock(text=conversion.name)
            text = tx_block['text']
            if text in defined_texts:
                conversion['name_addr'] = defined_texts[text]
            else:
                if stream:
                    address = tell()
                conversion['name_addr'] = address
                defined_texts[text] = address
                tx_block.address = address
                if stream:
                    write(bytes(tx_block))
                else:
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
                if stream:
                    address = tell()
                conversion['unit_addr'] = address
                defined_texts[text] = address
                tx_block.address = address
                if stream:
                    write(bytes(tx_block))
                else:
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
                if stream:
                    address = tell()
                conversion['comment_addr'] = address
                defined_texts[text] = address
                tx_block.address = address
                if stream:
                    write(bytes(tx_block))
                else:
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
                if stream:
                    address = tell()
                conversion['formula_addr'] = address
                defined_texts[text] = address
                tx_block.address = address
                if stream:
                    write(bytes(tx_block))
                else:
                    address += tx_block['block_len']
                    blocks.append(tx_block)

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
                    temp = dict(item)
                    write_cc(item, defined_texts, blocks, stream=stream)
                    address = tell()
                    item.address = address
                    conversion[key] = address
                    write(bytes(item))
                    item.update(temp)
                else:

                    item.address = address
                    conversion[key] = address
                    address += item['block_len']
                    blocks.append(item)
                    address = write_cc(item, defined_texts, blocks, address)

    return address


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
    attachments : list
        list of file attachments
    channels_db : dict
        used for fast channel access by name; for each name key the value is a
        list of (group index, channel index) tuples
    file_comment : TextBlock
        file comment TextBlock
    file_history : list
        list of (FileHistory, TextBlock) pairs
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

    def __init__(self, name=None, memory='full', version='4.10', callback=None, queue=None):
        memory = validate_memory_argument(memory)
        self.groups = []
        self.header = None
        self.identification = None
        self.file_history = []
        self.name = name
        self.memory = memory
        self.channels_db = {}
        self.masters_db = {}
        self.attachments = []
        self._attachments_cache = {}
        self.file_comment = None
        self.events = []

        self._attachments_map = {}
        self._ch_map = {}
        self._master_channel_cache = {}
        self._master_channel_metadata = {}
        self._invalidation_cache = {}
        self._si_map = {}
        self._cc_map = {}
        self._cg_map = {}
        self._dbc_cache = {}

        self._tempfile = TemporaryFile()
        self._file = None

        self._read_fragment_size = 0
        self._write_fragment_size = 8 * 2**20
        self._use_display_names = False
        self._single_bit_uint_as_bool = False

        # make sure no appended block has the address 0
        self._tempfile.write(b'\0')

        self._callback = callback
        self.queue = queue

        if name:
            self._file = open(self.name, 'rb')
            self._read()

        else:
            version = validate_version_argument(version)
            self.header = HeaderBlock()
            self.identification = FileIdentificationBlock(version=version)
            self.version = version

    def _check_finalised(self):
        flags = self.identification['unfinalized_standard_flags']
        if flags & 1:
            message = ('Unfinalised file {}:'
                       'Update of cycle counters for CG/CA blocks required')
            message = message.format(self.name)
            logger.warning(message)
        elif flags & 1 << 1:
            message = ('Unfinalised file {}:'
                       'Update of cycle counters for SR blocks required')
            message = message.format(self.name)
            logger.warning(message)
        elif flags & 1 << 2:
            message = ('Unfinalised file {}:'
                       'Update of length for last DT block required')
            message = message.format(self.name)
            logger.warning(message)
        elif flags & 1 << 3:
            message = ('Unfinalised file {}:'
                       'Update of length for last RD block required')
            message = message.format(self.name)
            logger.warning(message)
        elif flags & 1 << 4:
            message = ('Unfinalised file {}:'
                       'Update of last DL block in each chained list'
                       'of DL blocks required')
            message = message.format(self.name)
            logger.warning(message)
        elif flags & 1 << 5:
            message = ('Unfinalised file {}:'
                       'Update of cg_data_bytes and cg_inval_bytes '
                       'in VLSD CG block required')
            message = message.format(self.name)
            logger.warning(message)
        elif flags & 1 << 6:
            message = ('Unfinalised file {}:'
                       'Update of offset values for VLSD channel required '
                       'in case a VLSD CG block is used')
            message = message.format(self.name)
            logger.warning(message)

    def _read(self):

        stream = self._file
        memory = self.memory
        dg_cntr = 0

        cg_count = count_channel_groups(stream, 4)
        if self._callback:
            self._callback(0, cg_count)
        current_cg_index = 0

        self.identification = FileIdentificationBlock(stream=stream)
        version = self.identification['version_str']
        self.version = version.decode('utf-8').strip(' \n\t\0')

        if self.version >= '4.10':
            self._check_finalised()

        self.header = HeaderBlock(address=0x40, stream=stream)

        # read file history
        fh_addr = self.header['file_history_addr']
        while fh_addr:
            history_block = FileHistory(
                address=fh_addr,
                stream=stream,
            )
            self.file_history.append(history_block)
            fh_addr = history_block['next_fh_addr']

        # read attachments
        at_addr = self.header['first_attachment_addr']
        index = 0
        while at_addr:
            at_block = AttachmentBlock(address=at_addr, stream=stream)
            self._attachments_map[at_addr] = index
            self.attachments.append(at_block)
            at_addr = at_block['next_at_addr']
            index += 1

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

                grp['channels'] = []
                grp['logging_channels'] = []
                grp['data_block'] = None
                grp['channel_dependencies'] = []
                grp['signal_data'] = []

                # read each channel group sequentially
                block = ChannelGroup(address=cg_addr, stream=stream)
                self._cg_map[cg_addr] = dg_cntr
                channel_group = grp['channel_group'] = block

                grp['record_size'] = cg_size

                if channel_group['flags'] & v4c.FLAG_CG_VLSD:
                    # VLDS flag
                    record_id = channel_group['record_id']
                    cg_size[record_id] = 0
                elif channel_group['flags'] & v4c.FLAG_CG_BUS_EVENT:
                    bus_type = channel_group.acq_source['bus_type']
                    if bus_type == v4c.BUS_TYPE_CAN:
                        message_name = channel_group.acq_name

                        if message_name == 'CAN_DataFrame':
                            # this is a raw CAN bus logging channel group
                            # it will be later processed to extract all
                            # signals to new groups (one group per CAN message)
                            grp['raw_can'] = True
                            channel_group['flags'] &= ~v4c.FLAG_CG_BUS_EVENT
                            channel_group['flags'] &= ~v4c.FLAG_CG_PLAIN_BUS_EVENT

                        elif message_name in (
                                'CAN_ErrorFrame',
                                'CAN_RemoteFrame'):
                            # for now ignore bus logging flag
                            channel_group['flags'] &= ~v4c.FLAG_CG_BUS_EVENT
                            channel_group['flags'] &= ~v4c.FLAG_CG_PLAIN_BUS_EVENT
                        else:
                            comment = channel_group.comment.replace(' xmlns="http://www.asam.net/mdf/v4"', '')
                            comment_xml = ET.fromstring(comment)
                            can_msg_type = comment_xml.find('.//TX').text
                            if can_msg_type is not None:
                                can_msg_type = can_msg_type.strip(' \t\r\n')
                            else:
                                can_msg_type = 'CAN_DataFrame'
                            if can_msg_type == 'CAN_DataFrame':
                                common_properties = comment_xml.find(".//common_properties")
                                can_id = 1
                                message_id = -1
                                for e in common_properties:
                                    name = e.get('name')
                                    if name == 'MessageID':
                                        if e.get('ci') is not None:
                                            can_id = int(e.get('ci'))
                                        message_id = int(e.text)

                                if message_id > 0x80000000:
                                    message_id -= 0x80000000
                                grp['can_id'] = can_id
                                grp['message_name'] = message_name
                                grp['message_id'] = message_id

                            else:
                                message = 'Invalid bus logging channel group metadata: {}'.format(comment)
                                logger.warning(message)
                                channel_group['flags'] &= ~v4c.FLAG_CG_BUS_EVENT
                                channel_group['flags'] &= ~v4c.FLAG_CG_PLAIN_BUS_EVENT
                    else:
                        # only CAN bus logging is supported
                        channel_group['flags'] &= ~v4c.FLAG_CG_BUS_EVENT
                        channel_group['flags'] &= ~v4c.FLAG_CG_PLAIN_BUS_EVENT
                else:

                    samples_size = channel_group['samples_byte_nr']
                    inval_size = channel_group['invalidation_bytes_nr']
                    record_id = channel_group['record_id']
                    cg_size[record_id] = samples_size + inval_size

                if record_id_nr:
                    grp['sorted'] = False
                else:
                    grp['sorted'] = True

                data_group = DataGroup(address=dg_addr, stream=stream)
                grp['data_group'] = data_group

                # go to first channel of the current channel group
                ch_addr = channel_group['first_ch_addr']
                ch_cntr = 0
                neg_ch_cntr = -1

                # Read channels by walking recursively in the channel group
                # starting from the first channel
                self._read_channels(
                    ch_addr,
                    grp,
                    stream,
                    dg_cntr,
                    ch_cntr,
                    neg_ch_cntr,
                )

                cg_addr = channel_group['next_cg_addr']
                dg_cntr += 1

                current_cg_index += 1
                if self._callback:
                    self._callback(current_cg_index, cg_count)

                if self._terminate:
                    self.close()
                    return

                new_groups.append(grp)

            # store channel groups record sizes dict in each
            # new group data belong to the initial unsorted group, and add
            # the key 'sorted' with the value False to use a flag;
            # this is used later if memory is 'low' or 'minimum'

            if memory == 'full':
                grp['data_location'] = v4c.LOCATION_MEMORY
                dat_addr = group['data_block_addr']

                if record_id_nr == 0:
                    size = channel_group['samples_byte_nr']
                    size += channel_group['invalidation_bytes_nr']
                    size *= channel_group['cycles_nr']
                else:
                    size = 0
                    for gp in new_groups:
                        cg = gp['channel_group']
                        if cg['flags'] & v4c.FLAG_CG_VLSD:
                            total_vlsd_bytes = (cg['invalidation_bytes_nr'] << 32) + cg['samples_byte_nr']
                            size += total_vlsd_bytes + cg['cycles_nr'] * (record_id_nr + 4)
                        else:
                            size += (
                                (cg['samples_byte_nr']
                                 + record_id_nr
                                 + cg['invalidation_bytes_nr'])
                                * cg['cycles_nr']
                            )

                data = self._read_data_block(
                    address=dat_addr,
                    stream=stream,
                    size=size,
                )
                data = next(data)

                if record_id_nr == 0:
                    grp = new_groups[0]
                    grp['data_location'] = v4c.LOCATION_MEMORY
                    grp['data_block'] = DataBlock(data=data)

                    info = {
                        'data_block_addr': [],
                        'data_block_type': 0,
                        'data_size': [],
                        'data_block_size': [],
                        'param': 0,
                    }
                    grp.update(info)
                else:
                    cg_data = defaultdict(list)
                    if record_id_nr == 1:
                        fmt = '<B'
                    elif record_id_nr == 2:
                        fmt = '<H'
                    elif record_id_nr == 4:
                        fmt = '<I'
                    elif record_id_nr == 8:
                        fmt = '<Q'
                    else:
                        message = "invalid record id size {}"
                        raise MdfException(message.format(record_id_nr))

                    i = 0
                    while i < size:
                        # print(fmt, dg_cntr, len(data), size)
                        rec_id = unpack(fmt, data[i: i+record_id_nr])[0]
                        # skip record id
                        i += record_id_nr
                        rec_size = cg_size[rec_id]
                        if rec_size:
                            rec_data = data[i: i + rec_size]
                            cg_data[rec_id].append(rec_data)
                        else:
                            rec_size = unpack('<I', data[i: i + 4])[0]
                            rec_data = data[i: i + rec_size + 4]
                            cg_data[rec_id].append(rec_data)
                            i += 4
                        i += rec_size
                    for grp in new_groups:
                        grp['data_location'] = v4c.LOCATION_MEMORY
                        record_id = grp['channel_group']['record_id']
                        data = b''.join(cg_data[record_id])
                        grp['channel_group']['record_id'] = 1
                        grp['data_block'] = DataBlock(data=data)

                        info = {
                            'data_block_addr': [],
                            'data_block_type': 0,
                            'data_size': [],
                            'data_block_size': [],
                            'param': 0,
                        }
                        grp.update(info)
            else:
                address = group['data_block_addr']

                info = {
                    'data_block_addr': [],
                    'data_block_type': 0,
                    'data_size': [],
                    'data_block_size': [],
                    'param': 0,
                }

                # for low and minimum options save each block's type,
                # address and size

                if address:
                    stream.seek(address)
                    id_string, _, block_len, __ = unpack(
                        v4c.FMT_COMMON,
                        stream.read(v4c.COMMON_SIZE),
                    )
                    # can be a DataBlock
                    if id_string == b'##DT':
                        size = block_len - 24
                        info['data_size'].append(size)
                        info['data_block_size'].append(size)
                        info['data_block_addr'].append(address + v4c.COMMON_SIZE)
                        info['data_block_type'] = v4c.DT_BLOCK
                    # or a DataZippedBlock
                    elif id_string == b'##DZ':
                        stream.seek(address)
                        temp = {}
                        (temp['id'],
                         temp['reserved0'],
                         temp['block_len'],
                         temp['links_nr'],
                         temp['original_type'],
                         temp['zip_type'],
                         temp['reserved1'],
                         temp['param'],
                         temp['original_size'],
                         temp['zip_size'],) = unpack(
                            v4c.FMT_DZ_COMMON,
                            stream.read(v4c.DZ_COMMON_SIZE),
                        )
                        info['data_size'].append(temp['original_size'])
                        info['data_block_size'].append(temp['zip_size'])
                        info['data_block_addr'].append(address + v4c.DZ_COMMON_SIZE)
                        if temp['zip_type'] == v4c.FLAG_DZ_DEFLATE:
                            info['data_block_type'] = v4c.DZ_BLOCK_DEFLATE
                        else:
                            info['data_block_type'] = v4c.DZ_BLOCK_TRANSPOSED
                            info['param'] = temp['param']

                    # or a DataList
                    elif id_string == b'##DL':
                        info['data_block_type'] = v4c.DT_BLOCK
                        while address:
                            dl = DataList(address=address, stream=stream)
                            for i in range(dl['data_block_nr']):
                                addr = dl['data_block_addr{}'.format(i)]
                                info['data_block_addr'].append(addr + v4c.COMMON_SIZE)
                                stream.seek(addr+8)
                                size = unpack('<Q', stream.read(8))[0] - 24
                                info['data_size'].append(size)
                                info['data_block_size'].append(size)
                            address = dl['next_dl_addr']
                    # or a header list
                    elif id_string == b'##HL':
                        hl = HeaderList(address=address, stream=stream)
                        if hl['zip_type'] == v4c.FLAG_DZ_DEFLATE:
                            info['data_block_type'] = v4c.DZ_BLOCK_DEFLATE
                        else:
                            info['data_block_type'] = v4c.DZ_BLOCK_TRANSPOSED

                        address = hl['first_dl_addr']
                        while address:
                            dl = DataList(address=address, stream=stream)
                            for i in range(dl['data_block_nr']):
                                addr = dl['data_block_addr{}'.format(i)]
                                info['data_block_addr'].append(addr + v4c.DZ_COMMON_SIZE)
                                stream.seek(addr + 28)
                                param, size, zip_size = unpack(
                                    '<I2Q',
                                    stream.read(20),
                                )
                                info['data_size'].append(size)
                                info['data_block_size'].append(zip_size)
                                info['param'] = param

                            address = dl['next_dl_addr']

                for grp in new_groups:
                    grp['data_location'] = v4c.LOCATION_ORIGINAL_FILE
                    grp.update(info)

            self.groups.extend(new_groups)

            dg_addr = group['next_dg_addr']


        # all channels have been loaded so now we can link the
        # channel dependencies and load the signal data for VLSD channels
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
                sig_data_list = grp['signal_data']
                for i, signal_data_addr in enumerate(sig_data_list):

                    sig_data_list[i] = self._load_signal_data(
                        address=signal_data_addr,
                        stream=stream,
                    )

        # append indexes of groups that contain raw CAN bus logging and
        # store signals and metadata that will be used to create the new
        # groups.
        raw_can = []
        processed_can = []
        for i, group in enumerate(self.groups):
            if group.get('raw_can', False):
                can_ids = self.get('CAN_DataFrame.ID', group=i)
                all_can_ids = sorted(set(can_ids.samples))
                payload = self.get('CAN_DataFrame.DataBytes', group=i, samples_only=True)
                attachment, at_name = self.get('CAN_DataFrame', group=i).attachment

                if not at_name.lower().endswith(('dbc', 'arxml')) or not attachment:
                    message = 'Expected .dbc or .arxml file as CAN channel attachment but got "{}"'.format(at_name)
                    logger.warning(message)
                    grp['channel_group']['flags'] &= ~v4c.FLAG_CG_BUS_EVENT
                else:
                    raw_can.append(i)
                    import_type = 'dbc' if at_name.lower().endswith('dbc') else 'arxml'
                    db = loads(
                        attachment.decode('utf-8'),
                        importType=import_type,
                        key='db',
                    )['db']

                    board_units = set(bu.name for bu in db.boardUnits)

                    cg_source = group['channel_group'].acq_source

                    for message_id in all_can_ids:
                        sigs = []
                        can_msg = db.frameById(message_id)

                        for transmitter in can_msg.transmitter:
                            if transmitter in board_units:
                                break
                        else:
                            transmitter = ''
                        message_name = can_msg.name

                        source = SignalSource(
                            transmitter,
                            can_msg.name,
                            '',
                            v4c.SOURCE_BUS,
                            v4c.BUS_TYPE_CAN,
                        )

                        idx = argwhere(can_ids.samples == message_id).flatten()
                        data = payload[idx]
                        t = can_ids.timestamps[idx].copy()

                        for signal in sorted(can_msg.signals, key=lambda x: x.name):
                            # TODO : use grp['logging_channels'] instead of get_can_data
                            sig_vals = self._get_can_data(data, signal)
                            conversion = ChannelConversion(
                                a=signal.factor,
                                b=signal.offset,
                                conversion_type=v4c.CONVERSION_TYPE_LIN,
                            )
                            conversion.unit = signal.unit or ''
                            sigs.append(
                                Signal(
                                    sig_vals,
                                    t,
                                    name=signal.name,
                                    conversion=conversion,
                                    source=source,
                                    unit=signal.unit,
                                    raw=True,
                                )
                            )
                        processed_can.append(
                            [sigs, message_id, message_name, cg_source]
                        )

        # delete the groups that contain raw CAN bus logging and also
        # delete the channel entries from the channels_db. Update data group
        # index for the remaining channel entries. Append new data groups
        if raw_can:
            for index in reversed(raw_can):
                self.groups.pop(index)

            excluded_channels = []
            for name, db_entry in self.channels_db.items():
                new_entry = []
                for i, entry in enumerate(db_entry):
                    new_group_index = entry[0]
                    if new_group_index in raw_can:
                        continue
                    for index in raw_can:
                        if new_group_index > index:
                            new_group_index += 1
                        else:
                            break
                    new_entry.append((new_group_index, entry[1]))
                if new_entry:
                    self.channels_db[name] = new_entry
                else:
                    excluded_channels.append(name)
            for name in excluded_channels:
                del self.channels_db[name]

            for sigs, message_id, message_name, cg_source in processed_can:
                self.append(
                    sigs,
                    'Extracted from raw CAN bus logging',
                    common_timebase=True,
                )
                group = self.groups[-1]
                group['channel_group'].acq_source = cg_source
                group['data_group'].comment = 'From message {}="{}"'.format(
                    hex(message_id),
                    message_name,
                )

        # read events
        addr = self.header['first_event_addr']
        ev_map = {}
        event_index = 0
        while addr:
            event = EventBlock(address=addr, stream=stream)
            event.update_references(
                self._ch_map,
                self._cg_map,
            )
            self.events.append(event)
            ev_map[addr] = event_index
            event_index += 1

            addr = event['next_ev_addr']

        for event in self.events:
            addr = event['parent_ev_addr']
            if addr:
                event.parent = ev_map[addr]

            addr = event['range_start_ev_addr']
            if addr:
                event.range_start = ev_map[addr]

        if self.memory == 'full':
            self._file.close()

        self._si_map.clear()
        self._ch_map.clear()
        self._cc_map.clear()
        self._master_channel_cache.clear()

        self.progress = cg_count, cg_count

    def _read_channels(
            self,
            ch_addr,
            grp,
            stream,
            dg_cntr,
            ch_cntr,
            neg_ch_cntr,
            channel_composition=False):

        memory = self.memory
        channels = grp['channels']
        composition = []
        while ch_addr:
            # read channel block and create channel object
            if memory == 'minimum':
                channel = Channel(
                    address=ch_addr,
                    stream=stream,
                    cc_map=self._cc_map,
                    si_map=self._si_map,
                    load_metadata=False,
                    at_map=self._attachments_map,
                )
                value = ch_addr
                name = get_text_v4(
                    address=channel['name_addr'],
                    stream=stream,
                )
                comment = get_text_v4(
                    address=channel['comment_addr'],
                    stream=stream,
                ).replace(' xmlns="http://www.asam.net/mdf/v4"', '')

                if comment.startswith('<CNcomment'):
                    try:
                        display_name = ET.fromstring(comment).find('.//names/display')
                        if display_name is not None:
                            display_name = display_name.text
                    except UnicodeEncodeError:
                        display_name = ''
                else:
                    display_name = ''

            else:
                channel = Channel(
                    address=ch_addr,
                    stream=stream,
                    cc_map=self._cc_map,
                    si_map=self._si_map,
                    at_map=self._attachments_map,
                )
                value = channel
                display_name = channel.display_name
                name = channel.name

            self._ch_map[ch_addr] = (ch_cntr, dg_cntr)

            channels.append(value)
            if channel_composition:
                composition.append(
                    (ch_cntr, dg_cntr)
                )

            if display_name:
                if display_name not in self.channels_db:
                    self.channels_db[display_name] = []
                self.channels_db[display_name].append((dg_cntr, ch_cntr))

            # signal data
            address = channel['data_block_addr']
            grp['signal_data'].append(address)

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

            if channel['channel_type'] in MASTER_CHANNELS:
                self.masters_db[dg_cntr] = ch_cntr

            ch_cntr += 1

            if channel['component_addr']:
                # check if it is a CABLOCK or CNBLOCK
                stream.seek(channel['component_addr'])
                blk_id = stream.read(4)
                if blk_id == b'##CN':
                    index = ch_cntr - 1
                    grp['channel_dependencies'].append(None)
                    ch_cntr, neg_ch_cntr, ret_composition = self._read_channels(
                        channel['component_addr'],
                        grp,
                        stream,
                        dg_cntr,
                        ch_cntr,
                        neg_ch_cntr,
                        True,
                    )
                    grp['channel_dependencies'][index] = ret_composition

                    if grp['channel_group']['flags'] & v4c.FLAG_CG_BUS_EVENT and \
                            grp['channel_group']['flags'] & v4c.FLAG_CG_PLAIN_BUS_EVENT:
                        attachment_addr = self._attachments_map[channel['attachment_0_addr']]
                        if attachment_addr not in self._dbc_cache:
                            attachment, at_name = self.extract_attachment(index=attachment_addr)
                            if not at_name.lower().endswith(('dbc', 'arxml')) or not attachment:
                                message = 'Expected .dbc or .arxml file as CAN channel attachment but got "{}"'.format(at_name)
                                logger.warning(message)
                                grp['channel_group']['flags'] &= ~v4c.FLAG_CG_BUS_EVENT
                            else:
                                import_type = 'dbc' if at_name.lower().endswith('dbc') else 'arxml'
                                try:
                                    attachment_string = attachment.decode('utf-8')
                                    self._dbc_cache[attachment_addr] = \
                                        loads(
                                            attachment_string,
                                            importType=import_type,
                                            key='db',
                                        )['db']
                                except UnicodeDecodeError:
                                    try:
                                        from chardet import detect
                                        encoding = detect(attachment)['encoding']
                                        attachment_string = attachment.decode(encoding)
                                        self._dbc_cache[attachment_addr] = \
                                        loads(
                                            attachment_string,
                                            importType=import_type,
                                            key='db',
                                            encoding=encoding,
                                        )['db']
                                    except ImportError:
                                        message = (
                                            'Unicode exception occured while processing the database '
                                            'attachment "{}" and "chardet" package is '
                                            'not installed. Mdf version 4 expects "utf-8" '
                                            'strings and this package may detect if a different'
                                            ' encoding was used'
                                        ).format(at_name)
                                        logger.warning(message)
                                        grp['channel_group']['flags'] &= ~v4c.FLAG_CG_BUS_EVENT

                        if grp['channel_group']['flags'] & v4c.FLAG_CG_BUS_EVENT:

                            # here we make available multiple ways to refer to
                            # CAN signals by using fake negative indexes for
                            # the channel entries in the channels_db

                            grp['dbc_addr'] = attachment_addr

                            message_id = grp['message_id']
                            message_name = grp['message_name']
                            can_id = grp['can_id']

                            can_msg = self._dbc_cache[attachment_addr].frameById(message_id)
                            can_msg_name = can_msg.name

                            for entry in self.channels_db['CAN_DataFrame.DataBytes']:
                                if entry[0] == dg_cntr:
                                    index = entry[1]
                                    break

                            payload = channels[index]
                            if self.memory == 'minimum':
                                payload = Channel(
                                    stream=stream,
                                    address=payload,
                                )

                            logging_channels = grp['logging_channels']

                            for signal in can_msg.signals:
                                signal_name = signal.name

                                # 0 - name
                                # 1 - message_name.name
                                # 2 - can_id.message_name.name
                                # 3 - can_msg_name.name
                                # 4 - can_id.can_msg_name.name

                                name_ = signal_name
                                little_endian = True if signal.is_little_endian else False
                                signed = signal.is_signed
                                s_type = info_to_datatype_v4(signed, little_endian)
                                bit_offset = signal.startbit % 8
                                byte_offset = signal.startbit // 8
                                bit_count = signal.signalsize
                                comment = signal.comment or ''

                                if (signal.factor, signal.offset) != (1, 0):
                                    conversion = ChannelConversion(
                                        a=signal.factor,
                                        b=signal.offset,
                                        conversion_type=v4c.CONVERSION_TYPE_LIN,
                                    )
                                    conversion.unit = signal.unit or ''
                                else:
                                    conversion = None

                                kargs = {
                                    'channel_type': v4c.CHANNEL_TYPE_VALUE,
                                    'data_type': s_type,
                                    'sync_type': payload['sync_type'],
                                    'byte_offset': byte_offset + payload['byte_offset'],
                                    'bit_offset': bit_offset,
                                    'bit_count': bit_count,
                                    'min_raw_value': 0,
                                    'max_raw_value': 0,
                                    'lower_limit': 0,
                                    'upper_limit': 0,
                                    'flags': 0,
                                }

                                log_channel = Channel(**kargs)
                                log_channel.name = name_
                                log_channel.comment = comment
                                log_channel.source = deepcopy(channel.source)
                                log_channel.conversion = conversion
                                log_channel.unit = signal.unit or ''

                                logging_channels.append(log_channel)

                                if name_ not in self.channels_db:
                                    self.channels_db[name_] = []
                                self.channels_db[name_].append((dg_cntr, neg_ch_cntr))

                                name_ = '{}.{}'.format(message_name, signal_name)
                                if name_ not in self.channels_db:
                                    self.channels_db[name_] = []
                                self.channels_db[name_].append((dg_cntr, neg_ch_cntr))

                                name_ = 'CAN{}.{}.{}'.format(can_id, message_name, signal_name)
                                if name_ not in self.channels_db:
                                    self.channels_db[name_] = []
                                self.channels_db[name_].append((dg_cntr, neg_ch_cntr))

                                name_ = '{}.{}'.format(can_msg_name, signal_name)
                                if name_ not in self.channels_db:
                                    self.channels_db[name_] = []
                                self.channels_db[name_].append((dg_cntr, neg_ch_cntr))

                                name_ = 'CAN{}.{}.{}'.format(can_id, can_msg_name, signal_name)
                                if name_ not in self.channels_db:
                                    self.channels_db[name_] = []
                                self.channels_db[name_].append((dg_cntr, neg_ch_cntr))

                                neg_ch_cntr -= 1

                            grp['channel_group']['flags'] &= ~v4c.FLAG_CG_PLAIN_BUS_EVENT

                else:
                    # only channel arrays with storage=CN_TEMPLATE are
                    # supported so far
                    ca_block = ChannelArrayBlock(
                        address=channel['component_addr'],
                        stream=stream,
                    )
                    if ca_block['storage'] != v4c.CA_STORAGE_TYPE_CN_TEMPLATE:
                        logger.warning('Only CN template arrays are supported')
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

        return ch_cntr, neg_ch_cntr, composition

    def _read_data_block(self, address, stream, size=-1):
        """read and aggregate data blocks for a given data group

        Returns
        -------
        data : bytes
            aggregated raw data
        """
        if address:
            stream.seek(address)
            id_string = stream.read(4)
            # can be a DataBlock
            if id_string == b'##DT':
                data = DataBlock(address=address, stream=stream)
                data = data['data']
                yield data
            # or a DataZippedBlock
            elif id_string == b'##DZ':
                data = DataZippedBlock(address=address, stream=stream)
                data = data['data']
                yield data
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
                            stream.seek(addr)
                            id_string = stream.read(4)
                            if id_string == b'##DT':
                                _, dim, __ = unpack('<4s2Q', stream.read(20))
                                dim -= 24
                                position += stream.readinto(
                                    view[position: position+dim]
                                )
                            elif id_string == b'##DZ':
                                block = DataZippedBlock(
                                    stream=stream,
                                    address=addr,
                                )
                                uncompressed_size = block['original_size']
                                view[position: position+uncompressed_size] = block['data']
                                position += uncompressed_size
                        address = dl['next_dl_addr']
                    yield data

                else:
                    while address:
                        dl = DataList(address=address, stream=stream)
                        for i in range(dl['links_nr'] - 1):
                            addr = dl['data_block_addr{}'.format(i)]
                            stream.seek(addr)
                            id_string = stream.read(4)
                            if id_string == b'##DT':
                                block = DataBlock(stream=stream, address=addr)
                                yield block['data']
                            elif id_string == b'##DZ':
                                block = DataZippedBlock(
                                    stream=stream,
                                    address=addr,
                                )
                                yield block['data']
                            elif id_string == b'##DL':
                                for data in self._read_data_block(
                                        address=addr,
                                        stream=stream):
                                    yield data
                        address = dl['next_dl_addr']

            # or a header list
            elif id_string == b'##HL':
                hl = HeaderList(address=address, stream=stream)
                for data in self._read_data_block(
                        address=hl['first_dl_addr'],
                        stream=stream,
                        size=size):
                    yield data
        else:
            yield b''

    def _load_signal_data(self, address=None, stream=None, group=None, index=None):
        """ this method is used to get the channel signal data, usually for
        VLSD channels

        Parameters
        ----------
        address : int
            address of refrerenced block
        stream : handle
            file IO stream handle

        Returns
        -------
        data : bytes
            signal data bytes

        """

        if address == 0:
            data = b''

        elif address is not None and stream is not None:
            stream.seek(address)
            blk_id = stream.read(4)
            if blk_id == b'##SD':
                data = SignalDataBlock(address=address, stream=stream)
                data = data['data']
            elif blk_id == b'##DZ':
                data = DataZippedBlock(address=address, stream=stream)
                data = data['data']
            elif blk_id == b'##CG':
                group = self.groups[self._cg_map[address]]
                data = b''.join(fragment[0] for fragment in self._load_group_data(group))
            elif blk_id == b'##DL':
                data = []
                while address:
                    # the data list will contain only links to SDBLOCK's
                    data_list = DataList(address=address, stream=stream)
                    nr = data_list['links_nr']
                    # aggregate data from all SDBLOCK
                    for i in range(nr - 1):
                        addr = data_list['data_block_addr{}'.format(i)]
                        stream.seek(addr)
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
                            logger.warning(message)
                            return b''
                    address = data_list['next_dl_addr']
                data = b''.join(data)
            elif blk_id == b'##CN':
                data = b''
            elif blk_id == b'##HL':
                hl = HeaderList(address=address, stream=stream)

                data = self._load_signal_data(
                    address=hl['first_dl_addr'],
                    stream=stream,
                    group=group,
                    index=index,
                )
            else:
                message = ('Expected CG, SD, DL, DZ or CN block at {} '
                           'but found id="{}"')
                message = message.format(hex(address), blk_id)
                logger.warning(message)
                data = b''

        elif group is not None and index is not None:
            if group['data_location'] == v4c.LOCATION_ORIGINAL_FILE:
                data = self._load_signal_data(
                    address=group['signal_data'][index],
                    stream=self._file,
                )
            elif group['data_location'] == v4c.LOCATION_MEMORY:
                data = group['signal_data'][index]
            else:
                data = []
                stream = self._tempfile
                if group['signal_data'][index]:
                    for addr, size in zip(
                            group['signal_data'][index],
                            group['signal_data_size'][index]):
                        if not size:
                            continue
                        stream.seek(addr)
                        data.append(stream.read(size))
                data = b''.join(data)
        else:
            data = b''

        return data

    def _load_group_data(self, group):
        """ get group's data block bytes """
        offset = 0
        if self.memory == 'full':
            yield group['data_block']['data'], offset
        else:
            data_group = group['data_group']
            channel_group = group['channel_group']

            if group['data_location'] == v4c.LOCATION_ORIGINAL_FILE:
                stream = self._file
            else:
                stream = self._tempfile

            block_type = group['data_block_type']
            param = group['param']

            if not group['sorted']:
                cg_size = group['record_size']
                record_id = channel_group['record_id']
                if data_group['record_id_len'] <= 2:
                    record_id_nr = data_group['record_id_len']
                else:
                    record_id_nr = 0
            else:
                samples_size = (
                    channel_group['samples_byte_nr']
                    + channel_group['invalidation_bytes_nr']
                )

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

            if group['data_block_addr']:
                blocks = zip(
                    group['data_block_addr'],
                    group['data_size'],
                    group['data_block_size'],
                )
                if PYVERSION == 2:
                    blocks = iter(blocks)

                if block_type == v4c.DT_BLOCK and group['sorted']:
                    cur_size = 0
                    current_address = 0
                    data = []

                    while True:
                        try:
                            address, size, block_size = next(blocks)
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
                    if data:
                        yield b''.join(data), offset
                else:
                    for (address, size, block_size) in blocks:

                        stream.seek(address)
                        data = stream.read(block_size)

                        if block_type == v4c.DZ_BLOCK_DEFLATE:
                            data = decompress(data)

                        elif block_type == v4c.DZ_BLOCK_TRANSPOSED:
                            data = decompress(data)
                            cols = param
                            lines = size // cols

                            nd = fromstring(data[:lines * cols], dtype=uint8)
                            nd = nd.reshape((cols, lines))
                            data = nd.T.tostring() + data[lines * cols:]

                        if not group['sorted']:
                            rec_data = []

                            cg_size = group['record_size']
                            record_id = channel_group['record_id']
                            record_id_nr = data_group['record_id_len']

                            if record_id_nr == 1:
                                fmt = '<B'
                            elif record_id_nr == 2:
                                fmt = '<H'
                            elif record_id_nr == 4:
                                fmt = '<I'
                            elif record_id_nr == 8:
                                fmt = '<Q'
                            else:
                                message = "invalid record id size {}"
                                message = message.format(record_id_nr)
                                raise MdfException(message)

                            i = 0
                            size = len(data)
                            while i < size:
                                rec_id = unpack(fmt, data[i: i+record_id_nr])[0]
                                # skip record id
                                i += record_id_nr
                                rec_size = cg_size[rec_id]
                                if rec_size:
                                    if rec_id == record_id:
                                        rec_data.append(data[i: i + rec_size])
                                else:
                                    rec_size = unpack('<I', data[i: i + 4])[0]
                                    if rec_id == record_id:
                                        rec_data.append(data[i: i + 4 + rec_size])
                                    i += 4
                                i += rec_size
                            rec_data = b''.join(rec_data)
                            size = len(rec_data)
                            yield rec_data, offset
                            offset += size
                        else:
                            yield data, offset
                            offset += block_size
            else:
                yield b'', offset

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
        try:
            parents, dtypes = group['parents'], group['types']
        except KeyError:

            grp = group
            stream = self._file
            memory = self.memory
            channel_group = grp['channel_group']
            if memory == 'minimum':
                channels = [
                    Channel(
                        address=ch_addr,
                        stream=stream,
                        cc_map=self._cc_map,
                        si_map=self._si_map,
                        load_metadata=False,
                    )
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

            neg_index = -1

            sortedchannels = sorted(enumerate(channels), key=lambda i: i[1])
            for original_index, new_ch in sortedchannels:

                start_offset = new_ch['byte_offset']
                bit_offset = new_ch['bit_offset']
                data_type = new_ch['data_type']
                bit_count = new_ch['bit_count']
                ch_type = new_ch['channel_type']
                dependency_list = grp['channel_dependencies'][original_index]
                if memory == 'minimum':
                    name = get_text_v4(
                        address=new_ch['name_addr'],
                        stream=stream,
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
                            if data_type not in (
                                    v4c.DATA_TYPE_BYTEARRAY,
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
                                dtype_pair = name, get_fmt_v4(data_type, bit_count, ch_type)
                                types.append(dtype_pair)
                                parents[original_index] = name, bit_offset
                            else:
                                next_byte_aligned_position = parent_start_offset

                            current_parent = name
                        else:
                            if isinstance(dependency_list[0], ChannelArrayBlock):
                                ca_block = dependency_list[0]

                                # check if there are byte gaps in the record
                                gap = start_offset - next_byte_aligned_position
                                if gap:
                                    dtype_pair = '', 'a{}'.format(gap)
                                    types.append(dtype_pair)

                                size = max(bit_count >> 3, 1)
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

                                dtype_pair = name, get_fmt_v4(data_type, bit_count), shape
                                types.append(dtype_pair)

                                current_parent = name
                                next_byte_aligned_position = start_offset + size * dim
                                parents[original_index] = name, 0

                            else:
                                parents[original_index] = None, None
                                if channel_group['flags'] & v4c.FLAG_CG_BUS_EVENT:
                                    for logging_channel in grp['logging_channels']:
                                        parents[neg_index] = 'CAN_DataFrame.DataBytes', logging_channel['bit_offset']
                                        neg_index -= 1

                    # virtual channels do not have bytes in the record
                    else:
                        parents[original_index] = None, None

                else:
                    max_overlapping_size = (next_byte_aligned_position - start_offset) * 8
                    needed_size = bit_offset + bit_count
                    if max_overlapping_size >= needed_size:
                        parents[original_index] = (
                            current_parent,
                            ((start_offset - parent_start_offset) << 3) + bit_offset,
                        )
                if next_byte_aligned_position > record_size:
                    break

            gap = record_size - next_byte_aligned_position
            if gap > 0:
                dtype_pair = '', 'a{}'.format(gap)
                types.append(dtype_pair)

            dtype_pair = 'invalidation_bytes', '<u1', invalidation_bytes_nr
            types.append(dtype_pair)
            if PYVERSION == 2:
                types = fix_dtype_fields(types)

            dtypes = dtype(types)

        return parents, dtypes

    def _append_structure_composition(
            self, grp, signal, field_names, offset,
            dg_cntr, ch_cntr, parents, defined_texts, cc_map, si_map):

        fields = []
        types = []

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

        memory = self.memory
        file = self._tempfile
        seek = file.seek
        seek(0, 2)

        gp = grp
        gp_sdata = gp['signal_data']
        gp_sdata_size = gp['signal_data_size']
        gp_channels = gp['channels']
        gp_dep = gp['channel_dependencies']

        name = signal.name
        names = signal.samples.dtype.names

        field_name = get_unique_name(field_names, name)
        field_names.add(field_name)

        # first we add the structure channel

        if signal.attachment:
            at_data, at_name = signal.attachment
            attachment_addr = self.attach(
                at_data,
                at_name,
                mime='application/x-dbc',
            )
        else:
            attachment_addr = 0

        # add channel block
        kargs = {
            'channel_type': v4c.CHANNEL_TYPE_VALUE,
            'bit_count': signal.samples.dtype.itemsize * 8,
            'byte_offset': offset,
            'bit_offset': 0,
            'data_type': v4c.DATA_TYPE_BYTEARRAY,
            'min_raw_value': 0,
            'max_raw_value': 0,
            'lower_limit': 0,
            'upper_limit': 0,
            'flags': 0,
            'precision': 255,
        }
        if attachment_addr:
            kargs['attachment_0_addr'] = attachment_addr
            kargs['flags'] |= v4c.FLAG_CN_BUS_EVENT
        ch = Channel(**kargs)
        ch.name = name
        ch.unit = signal.unit
        ch.comment = signal.comment
        ch.display_name = signal.display_name

        # source for channel
        if signal.source:
            source = signal.source
            new_source = SourceInformation(
                source_type=signal.source.source_type,
                bus_type=signal.source.bus_type,
            )
            new_source.name = source.name
            new_source.path = source.path
            new_source.comment = source.comment

            ch.source = new_source

        if memory != 'minimum':
            gp_channels.append(ch)
            struct_self = ch_cntr, dg_cntr
        else:
            ch.to_stream(file, defined_texts, cc_map, si_map)
            gp_channels.append(ch.address)
            struct_self = ch_cntr, dg_cntr

        gp_sdata.append(None)
        gp_sdata_size.append(0)
        if name not in self.channels_db:
            self.channels_db[name] = []
        self.channels_db[name].append((dg_cntr, ch_cntr))

        # update the parents as well
        parents[ch_cntr] = name, 0

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

        dep_list = []
        gp_dep.append(dep_list)

        # then we add the fields

        for name in names:
            field_name = get_unique_name(field_names, name)
            field_names.add(field_name)

            samples = signal.samples[name]
            fld_names = samples.dtype.names

            if fld_names is None:
                sig_type = v4c.SIGNAL_TYPE_SCALAR
                if samples.dtype.kind in 'SV':
                    sig_type = v4c.SIGNAL_TYPE_STRING
            else:
                if fld_names in (canopen_time_fields, canopen_date_fields):
                    sig_type = v4c.SIGNAL_TYPE_CANOPEN
                elif fld_names[0] != name:
                    sig_type = v4c.SIGNAL_TYPE_STRUCTURE_COMPOSITION
                else:
                    sig_type = v4c.SIGNAL_TYPE_ARRAY

            if sig_type == v4c.SIGNAL_TYPE_SCALAR:

                s_type, s_size = fmt_to_datatype_v4(
                    samples.dtype,
                    samples.shape,
                )
                byte_size = s_size >> 3

                fields.append(samples)
                types.append((field_name, samples.dtype, samples.shape[1:]))

                # add channel block
                min_val, max_val = get_min_max(samples)
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
                    'precision': 255,
                }
                if min_val > max_val or s_type == v4c.DATA_TYPE_BYTEARRAY:
                    kargs['flags'] = v4c.FLAG_CN_PRECISION
                else:
                    kargs['flags'] = v4c.FLAG_PHY_RANGE_OK | v4c.FLAG_VAL_RANGE_OK
                if attachment_addr:
                    kargs['flags'] |= v4c.FLAG_CN_BUS_EVENT

                ch = Channel(**kargs)
                ch.name = name

                if memory != 'minimum':
                    gp_channels.append(ch)
                    dep_list.append(
                        (ch_cntr, dg_cntr)
                    )
                else:
                    ch.to_stream(file, defined_texts, cc_map, si_map)
                    gp_channels.append(ch.address)
                    dep_list.append(
                        (ch_cntr, dg_cntr)
                    )

                offset += byte_size

                gp_sdata.append(None)
                gp_sdata_size.append(0)
                if name not in self.channels_db:
                    self.channels_db[name] = []
                self.channels_db[name].append((dg_cntr, ch_cntr))

                # update the parents as well
                parents[ch_cntr] = field_name, 0

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
                gp_dep.append(None)

            elif sig_type == v4c.SIGNAL_TYPE_STRUCTURE_COMPOSITION:
                struct = Signal(
                    samples,
                    samples,
                    name=name,
                )
                offset, dg_cntr, ch_cntr, sub_structure, new_fields, new_types = self._append_structure_composition(
                    grp, struct, field_names, offset, dg_cntr, ch_cntr,
                    parents, defined_texts, cc_map, si_map,
                )
                dep_list.append(sub_structure)
                fields.extend(new_fields)
                types.extend(new_types)

        return offset, dg_cntr, ch_cntr, struct_self, fields, types

    def _get_not_byte_aligned_data(self, data, group, ch_nr):
        big_endian_types = (
            v4c.DATA_TYPE_UNSIGNED_MOTOROLA,
            v4c.DATA_TYPE_REAL_MOTOROLA,
            v4c.DATA_TYPE_SIGNED_MOTOROLA,
        )

        record_size = group['channel_group']['samples_byte_nr']

        if ch_nr >= 0:
            if self.memory == 'minimum':
                if group['data_location'] == v4c.LOCATION_ORIGINAL_FILE:
                    channel = Channel(
                        address=group['channels'][ch_nr],
                        stream=self._file,
                        load_metadata=False,
                    )
                else:
                    channel = Channel(
                        address=group['channels'][ch_nr],
                        stream=self._tempfile,
                        load_metadata=False,
                    )
            else:
                channel = group['channels'][ch_nr]
        else:
            channel = group['logging_channels'][-ch_nr-1]

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

        fmt = get_fmt_v4(channel['data_type'], bit_count)
        if size <= byte_count:
            if channel['data_type'] in big_endian_types:
                types = [
                    ('', 'a{}'.format(byte_count - size)),
                    ('vals', fmt),
                ]
            else:
                types = [
                    ('vals', fmt),
                    ('', 'a{}'.format(byte_count - size)),
                ]
        else:
            types = [('vals', fmt), ]

        vals = fromstring(vals, dtype=dtype(types))['vals']

        if channel['data_type'] in v4c.SIGNED_INT:
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
        suppress = True
        if name is None:
            if group is None or index is None:
                message = (
                    'Invalid arguments for channel selection: '
                    'must give "name" or, "group" and "index"'
                )
                raise MdfException(message)
            else:
                gp_nr, ch_nr = group, index
                if ch_nr >= 0:
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
                    if len(self.channels_db[name]) > 1 and not suppress:
                        message = (
                            'Multiple occurances for channel "{}". '
                            'Using first occurance from data group {}. '
                            'Provide both "group" and "index" arguments'
                            ' to select another data group'
                        )
                        message = message.format(name, gp_nr)
                        logger.warning(message)
                else:
                    if index is not None and index < 0:
                        gp_nr = group
                        ch_nr = index
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

    def get_valid_indexes(self, group_index, channel, fragment):
        """ get invalidation indexes for the channel

        Parameters
        ----------
        group_index : int
            group index
        channel : Channel
            channel object
        fragment : (bytes, int)
            (fragment bytes, fragment offset)

        Returns
        -------
        valid_indexes : iterable
            iterable of valid channel indexes; if all are valid `None` is
            returned

        """
        group = self.groups[group_index]
        dtypes = group['types']

        data_bytes, offset = fragment
        try:
            invalidation = self._invalidation_cache[(group_index, offset)]
        except KeyError:
            not_found = object()
            record = group.get('record', not_found)
            if record is not_found:
                if dtypes.itemsize:
                    record = fromstring(data_bytes, dtype=dtypes)
                else:
                    record = None

            invalidation = record['invalidation_bytes'].copy()
            self._invalidation_cache[(group_index, offset)] = invalidation

        ch_invalidation_pos = channel['pos_invalidation_bit']
        pos_byte, pos_offset = divmod(ch_invalidation_pos, 8)
        mask = 1 << pos_offset

        valid_indexes = array(
            [bytes_[pos_byte] & mask for bytes_ in invalidation]
        )
        valid_indexes = argwhere(valid_indexes == 0).flatten()

        return valid_indexes

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
            size hint of splitted data blocks, default 8MB; if the initial size is
            smaller, then no data list is used. The actual split size depends on
            the data groups' records size
        write_fragment_size : int
            size hint of splitted data blocks, default 8MB; if the initial size is
            smaller, then no data list is used. The actual split size depends on
            the data groups' records size
        use_display_names : bool
            use display name if available for the Signal's name returned by the get method

        """

        if read_fragment_size is not None:
            self._read_fragment_size = int(read_fragment_size)

        if write_fragment_size:
            self._write_fragment_size = int(write_fragment_size)

        if use_display_names is not None:
            self._use_display_names = bool(use_display_names)

        if single_bit_uint_as_bool is not None:
            self._single_bit_uint_as_bool = bool(single_bit_uint_as_bool)

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
        gp['signal_data'] = gp_sdata = []
        gp['signal_data_size'] = gp_sdata_size = []
        gp['channels'] = gp_channels = []
        gp['channel_dependencies'] = gp_dep = []
        gp['signal_types'] = gp_sig_types = []
        gp['logging_channels'] = []

        self.groups.append(gp)

        cycles_nr = len(t)
        fields = []
        types = []
        parents = {}
        ch_cntr = 0
        offset = 0
        field_names = set()

        defined_texts = {}
        si_map = {}
        cc_map = {}

        # setup all blocks related to the time master channel

        memory = self.memory
        file = self._tempfile
        write = file.write
        tell = file.tell
        seek = file.seek

        seek(0, 2)

        master_metadata = signals[0].master_metadata
        if master_metadata:
            time_name, sync_type = master_metadata
            if sync_type in (0, 1):
                time_unit = 's'
            elif sync_type == 2:
                time_unit = 'deg'
            elif sync_type == 3:
                time_unit = 'm'
            elif sync_type == 4:
                time_unit = 'index'
        else:
            time_name, sync_type = 'Time', v4c.SYNC_TYPE_TIME
            time_unit = 's'

        source_block = SourceInformation()
        source_block.name = source_block.path = source_info

        # time channel
        t_type, t_size = fmt_to_datatype_v4(
            t.dtype,
            t.shape,
        )
        kargs = {
            'channel_type': v4c.CHANNEL_TYPE_MASTER,
            'data_type': t_type,
            'sync_type': sync_type,
            'byte_offset': 0,
            'bit_offset': 0,
            'bit_count': t_size,
            'min_raw_value': t[0] if cycles_nr else 0,
            'max_raw_value': t[-1] if cycles_nr else 0,
            'lower_limit': t[0] if cycles_nr else 0,
            'upper_limit': t[-1] if cycles_nr else 0,
            'flags': v4c.FLAG_PHY_RANGE_OK | v4c.FLAG_VAL_RANGE_OK,
        }
        ch = Channel(**kargs)
        ch.unit = time_unit
        ch.name = time_name
        ch.source = source_block
        name = time_name
        if memory == 'minimum':
            ch.to_stream(file, defined_texts, cc_map, si_map)
            gp_channels.append(ch.address)
        else:
            gp_channels.append(ch)

        gp_sdata.append(None)
        gp_sdata_size.append(0)
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

        gp_sig_types.append(0)

        # check if the source is included in the channel name
        name = name.split('\\')
        if len(name) > 1:
            name = name[0]
            if name in self.channels_db:
                self.channels_db[name].append((dg_cntr, ch_cntr))
            else:
                self.channels_db[name] = []
                self.channels_db[name].append((dg_cntr, ch_cntr))

        for signal in signals:
            sig = signal
            names = sig.samples.dtype.names
            name = signal.name

            if names is None:
                sig_type = v4c.SIGNAL_TYPE_SCALAR
                if sig.samples.dtype.kind in 'SV':
                    sig_type = v4c.SIGNAL_TYPE_STRING
            else:
                if names in (canopen_time_fields, canopen_date_fields):
                    sig_type = v4c.SIGNAL_TYPE_CANOPEN
                elif names[0] != sig.name:
                    sig_type = v4c.SIGNAL_TYPE_STRUCTURE_COMPOSITION
                else:
                    sig_type = v4c.SIGNAL_TYPE_ARRAY

            gp_sig_types.append(sig_type)

            # first add the signals in the simple signal list
            if sig_type == v4c.SIGNAL_TYPE_SCALAR:

                # compute additional byte offset for large records size
                s_type, s_size = fmt_to_datatype_v4(
                    signal.samples.dtype,
                    signal.samples.shape,
                )

                byte_size = max(s_size // 8, 1)
                min_val, max_val = get_min_max(signal.samples)

                if signal.samples.dtype.kind == 'u' and signal.bit_count <= 4:
                    s_size = signal.bit_count

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
                }

                if min_val > max_val or s_type == v4c.DATA_TYPE_BYTEARRAY:
                    kargs['flags'] = 0
                else:
                    kargs['flags'] = v4c.FLAG_PHY_RANGE_OK | v4c.FLAG_VAL_RANGE_OK
                ch = Channel(**kargs)
                ch.name = name
                ch.unit = signal.unit
                ch.comment = signal.comment
                ch.display_name = signal.display_name

                # conversions for channel
                conversion = conversion_transfer(signal.conversion, version=4)
                if signal.raw:
                    ch.conversion = conversion

                # source for channel
                if signal.source:
                    source = signal.source
                    new_source = SourceInformation(
                        source_type=signal.source.source_type,
                        bus_type=signal.source.bus_type,
                    )
                    new_source.name = source.name
                    new_source.path = source.path
                    new_source.comment = source.comment

                    ch.source = new_source

                if memory != 'minimum':
                    gp_channels.append(ch)
                else:
                    ch.to_stream(file, defined_texts, cc_map, si_map)
                    gp_channels.append(ch.address)

                offset += byte_size

                gp_sdata.append(None)
                gp_sdata_size.append(0)
                if name not in self.channels_db:
                    self.channels_db[name] = []
                self.channels_db[name].append((dg_cntr, ch_cntr))

                # update the parents as well
                field_name = get_unique_name(field_names, name)
                parents[ch_cntr] = field_name, 0

                fields.append(signal.samples)
                if s_type == v4c.DATA_TYPE_BYTEARRAY:
                    types.append(
                        (field_name, signal.samples.dtype, signal.samples.shape[1:])
                    )
                else:
                    types.append(
                        (field_name, signal.samples.dtype)
                    )
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

                ch_cntr += 1

                # simple channels don't have channel dependencies
                gp_dep.append(None)

            elif sig_type == v4c.SIGNAL_TYPE_STRING:
                offsets = arange(
                    len(signal),
                    dtype=uint64,
                ) * (signal.samples.itemsize + 4)

                values = [
                    ones(len(signal), dtype=uint32) * signal.samples.itemsize,
                    signal.samples,
                ]

                types_ = [
                    ('', uint32),
                    ('', signal.samples.dtype),
                ]

                data = fromarrays(values, dtype=types_).tostring()

                if memory == 'full':
                    gp_sdata.append(data)
                    data_addr = 0
                else:
                    if data:
                        data_addr = tell()
                        gp_sdata.append([data_addr, ])
                        gp_sdata_size.append([len(data), ])
                        write(data)
                    else:
                        data_addr = 0
                        gp_sdata.append([])
                        gp_sdata_size.append([])

                # compute additional byte offset for large records size
                byte_size = 8
                kargs = {
                    'channel_type': v4c.CHANNEL_TYPE_VLSD,
                    'bit_count': 64,
                    'byte_offset': offset,
                    'bit_offset': 0,
                    'data_type': v4c.DATA_TYPE_STRING_UTF_8,
                    'min_raw_value':  0,
                    'max_raw_value': 0,
                    'lower_limit': 0,
                    'upper_limit': 0,
                    'flags': 0,
                    'data_block_addr': data_addr,
                }
                ch = Channel(**kargs)
                ch.name = name
                ch.unit = signal.unit
                ch.comment = signal.comment
                ch.display_name = signal.display_name

                # conversions for channel
                conversion = conversion_transfer(signal.conversion, version=4)
                if signal.raw:
                    ch.conversion = conversion

                # source for channel
                if signal.source:
                    source = signal.source
                    new_source = SourceInformation(
                        source_type=signal.source.source_type,
                        bus_type=signal.source.bus_type,
                    )
                    new_source.name = source.name
                    new_source.path = source.path
                    new_source.comment = source.comment

                    ch.source = new_source

                if memory != 'minimum':
                    gp_channels.append(ch)
                else:
                    ch.to_stream(file, defined_texts, cc_map, si_map)
                    gp_channels.append(ch.address)

                offset += byte_size

                if name not in self.channels_db:
                    self.channels_db[name] = []
                self.channels_db[name].append((dg_cntr, ch_cntr))

                # update the parents as well
                field_name = get_unique_name(field_names, name)
                parents[ch_cntr] = field_name, 0

                fields.append(offsets)
                types.append((field_name, uint64))
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

                ch_cntr += 1

                # simple channels don't have channel dependencies
                gp_dep.append(None)

            elif sig_type == v4c.SIGNAL_TYPE_CANOPEN:

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
                        if field == 'hour':
                            vals.append(signal.samples[field] + (signal.samples['summer_time'] << 7))
                        elif field == 'day':
                            vals.append(signal.samples[field] + (signal.samples['day_of_week'] << 4))
                        else:
                            vals.append(signal.samples[field])
                    vals = fromarrays(vals).tostring()

                    fields.append(frombuffer(vals, dtype='V7'))
                    types.append((field_name, 'V7'))
                    byte_size = 7
                    s_type = v4c.DATA_TYPE_CANOPEN_DATE

                s_size = byte_size << 3

                # there is no channel dependency
                gp_dep.append(None)

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
                }
                ch = Channel(**kargs)
                ch.name = name
                ch.unit = signal.unit
                ch.comment = signal.comment
                ch.display_name = signal.display_name

                # source for channel
                if signal.source:
                    source = signal.source
                    new_source = SourceInformation(
                        source_type=signal.source.source_type,
                        bus_type=signal.source.bus_type,
                    )
                    new_source.name = source.name
                    new_source.path = source.path
                    new_source.comment = source.comment

                    ch.source = new_source

                if memory != 'minimum':
                    gp_channels.append(ch)
                else:
                    ch.to_stream(file, defined_texts, cc_map, si_map)
                    gp_channels.append(ch.address)

                offset += byte_size

                if name in self.channels_db:
                    self.channels_db[name].append((dg_cntr, ch_cntr))
                else:
                    self.channels_db[name] = []
                    self.channels_db[name].append((dg_cntr, ch_cntr))

                # update the parents as well
                parents[ch_cntr] = field_name, 0

                if memory == 'full':
                    gp_sdata.append(None)
                    gp_sdata_size.append(0)
                else:
                    gp_sdata.append(0)
                    gp_sdata_size.append(0)

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

            elif sig_type == v4c.SIGNAL_TYPE_STRUCTURE_COMPOSITION:
                offset, dg_cntr, ch_cntr, struct_self, new_fields, new_types = self._append_structure_composition(
                    gp, signal, field_names,
                    offset, dg_cntr, ch_cntr,
                    parents, defined_texts, cc_map, si_map)
                fields.extend(new_fields)
                types.extend(new_types)

            else:
                # here we have channel arrays or mdf v3 channel dependencies
                samples = signal.samples[names[0]]
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
                s_type, s_size = fmt_to_datatype_v4(
                    samples.dtype,
                    samples.shape,
                    True,
                )

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
                }
                ch = Channel(**kargs)
                ch.name = name
                ch.unit = signal.unit
                ch.comment = signal.comment
                ch.display_name = signal.display_name

                # source for channel
                if signal.source:
                    source = signal.source
                    new_source = SourceInformation(
                        source_type=signal.source.source_type,
                        bus_type=signal.source.bus_type,
                    )
                    new_source.name = source.name
                    new_source.path = source.path
                    new_source.comment = source.comment

                    ch.source = new_source

                if memory != 'minimum':
                    gp_channels.append(ch)
                else:
                    ch.to_stream(file, defined_texts, cc_map, si_map)
                    gp_channels.append(ch.address)

                size = s_size >> 3
                for dim in shape:
                    size *= dim
                offset += size

                gp_sdata.append(None)
                gp_sdata_size.append(0)
                if name not in self.channels_db:
                    self.channels_db[name] = []
                self.channels_db[name].append((dg_cntr, ch_cntr))

                # update the parents as well
                parents[ch_cntr] = name, 0

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

                for name in names[1:]:
                    field_name = get_unique_name(field_names, name)
                    field_names.add(field_name)

                    samples = signal.samples[name]
                    shape = samples.shape[1:]
                    fields.append(samples)
                    types.append((field_name, samples.dtype, shape))

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
                    s_type, s_size = fmt_to_datatype_v4(
                        samples.dtype,
                        (),
                    )
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
                    }

                    ch = Channel(**kargs)
                    ch.name = name
                    ch.unit = signal.unit
                    ch.comment = signal.comment
                    ch.display_name = signal.display_name

                    if memory != 'minimum':
                        gp_channels.append(ch)
                    else:
                        ch.to_stream(file, defined_texts, cc_map, si_map)
                        gp_channels.append(ch.address)

                    parent_dep.referenced_channels.append((ch_cntr, dg_cntr))
                    for dim in shape:
                        byte_size *= dim
                    offset += byte_size

                    gp_sdata.append(None)
                    gp_sdata_size.append(0)
                    if name not in self.channels_db:
                        self.channels_db[name] = []
                    self.channels_db[name].append((dg_cntr, ch_cntr))

                    # update the parents as well
                    parents[ch_cntr] = field_name, 0

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

        # channel group
        kargs = {
            'cycles_nr': cycles_nr,
            'samples_byte_nr': offset,
        }
        gp['channel_group'] = ChannelGroup(**kargs)
        gp['size'] = cycles_nr * offset

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

                gp['data_block_type'] = v4c.DT_BLOCK
                gp['param'] = 0
                gp['data_size'] = []
                gp['data_block_size'] = []
                gp['data_block_addr'] = []

            else:
                if block:
                    data_address = self._tempfile.tell()
                    gp['data_location'] = v4c.LOCATION_TEMPORARY_FILE
                    gp['data_block'] = [data_address, ]
                    gp['data_group']['data_block_addr'] = data_address
                    size = len(block)
                    self._tempfile.write(block)
                    gp['data_block_type'] = v4c.DT_BLOCK
                    gp['param'] = 0
                    gp['data_size'] = [size, ]
                    gp['data_block_size'] = [size, ]
                    gp['data_block_addr'] = [data_address, ]
                else:
                    gp['data_location'] = v4c.LOCATION_TEMPORARY_FILE
                    gp['data_block'] = [0, ]
                    gp['data_group']['data_block_addr'] = 0
                    gp['data_block_type'] = v4c.DT_BLOCK
                    gp['param'] = 0
                    gp['data_size'] = [0, ]
                    gp['data_block_size'] = [0, ]
                    gp['data_block_addr'] = [0, ]

        except MemoryError:
            if memory == 'full':
                raise
            else:
                size = 0
                gp['data_location'] = v4c.LOCATION_TEMPORARY_FILE

                data_address = self._tempfile.tell()
                gp['data_group']['data_block_addr'] = data_address
                for sample in samples:
                    size += self._tempfile.write(sample.tostring())
                gp['data_block_type'] = v4c.DT_BLOCK
                gp['param'] = 0
                gp['data_size'] = [size, ]
                gp['data_block_size'] = [size, ]
                if size:
                    gp['data_block_addr'] = [data_address, ]
                else:
                    gp['data_block_addr'] = [0, ]

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
        gp = self.groups[index]
        if not signals:
            message = '"append" requires a non-empty list of Signal objects'
            raise MdfException(message)

        if gp['data_location'] == v4c.LOCATION_ORIGINAL_FILE:
            stream = self._file
        else:
            stream = self._tempfile

        canopen_time_fields = (
            'ms',
            'days',
        )

        fields = []
        types = []

        for i, (signal, sig_type) in enumerate(
                zip(signals, gp['signal_types'])):

            # first add the signals in the simple signal list
            if sig_type == v4c.SIGNAL_TYPE_SCALAR:

                fields.append(signal)
                if signal.shape[1:]:
                    types.append(('', signal.dtype, signal.shape[1:]))
                else:
                    types.append(('', signal.dtype))
                min_val, max_val = get_min_max(signal)
                if self.memory == 'minimum':
                    address = gp['channels'][i]
                    channel = Channel(
                        address=address,
                        stream=stream,
                        load_metadata=False,
                    )

                    update = False
                    if min_val < channel['min_raw_value']:
                        channel['min_raw_value'] = min_val
                        channel['lower_limit'] = min_val
                        update = True
                    if max_val > channel['max_raw_value']:
                        channel['max_raw_value'] = max_val
                        channel['upper_limit'] = max_val
                        update = True

                    if update:
                        stream.seek(address)
                        stream.write(bytes(channel))

                else:
                    channel = gp['channels'][i]
                    if min_val < channel['min_raw_value']:
                        channel['min_raw_value'] = min_val
                        channel['lower_limit'] = min_val
                    if max_val > channel['max_raw_value']:
                        channel['max_raw_value'] = max_val
                        channel['upper_limit'] = max_val

            elif sig_type == v4c.SIGNAL_TYPE_STRING:
                if self.memory == 'full':
                    data = gp['signal_data'][i]
                    cur_offset = len(data)
                else:
                    cur_offset = sum(gp['signal_data_size'][i])

                offsets = arange(len(signal), dtype=uint64) * (signal.itemsize + 4) + cur_offset
                values = [
                    ones(len(signal), dtype=uint32) * signal.itemsize,
                    signal,
                ]

                types_ = [
                    ('', uint32),
                    ('', signal.dtype),
                ]

                values = fromarrays(values, dtype=types_).tostring()

                if self.memory == 'full':
                    gp['signal_data'][i] = data + values
                else:
                    stream.seek(0, 2)
                    addr = stream.tell()
                    if values:
                        stream.write(values)
                        gp['signal_data'][i].append(addr)
                        gp['signal_data_size'][i].append(len(values))

                fields.append(offsets)
                types.append(('', uint64))

            elif sig_type == v4c.SIGNAL_TYPE_CANOPEN:
                names = signal.dtype.names

                if names == canopen_time_fields:

                    vals = signal.tostring()

                    fields.append(frombuffer(vals, dtype='V6'))
                    types.append(('', 'V6'))

                else:
                    vals = []
                    for field in ('ms', 'min', 'hour', 'day', 'month', 'year'):
                        vals.append(signal[field])
                    vals = fromarrays(vals).tostring()

                    fields.append(frombuffer(vals, dtype='V7'))
                    types.append(('', 'V7'))

            elif sig_type == v4c.SIGNAL_TYPE_STRUCTURE_COMPOSITION:
                names = signal.dtype.names
                for name in names:
                    samples = signal[name]

                    fields.append(samples)
                    types.append(('', samples.dtype))

            elif sig_type == v4c.SIGNAL_TYPE_ARRAY:
                names = signal.dtype.names

                samples = signal[names[0]]

                shape = samples.shape[1:]

                fields.append(samples)
                types.append(
                    ('', samples.dtype, shape)
                )

                for name in names[1:]:

                    samples = signal[name]
                    shape = samples.shape[1:]
                    fields.append(samples)
                    types.append(
                        ('', samples.dtype, shape)
                    )

        # data block
        if PYVERSION == 2:
            types = fix_dtype_fields(types)
        types = dtype(types)

        samples = fromarrays(fields, dtype=types).tostring()
        del fields
        del types

        if self.memory == 'full':
            samples = gp['data_block']['data'] + samples
            gp['data_block'] = DataBlock(data=samples)

            size = gp['data_block']['block_len'] - v4c.COMMON_SIZE

            record_size = gp['channel_group']['samples_byte_nr']
            record_size += gp['data_group']['record_id_len']
            gp['channel_group']['cycles_nr'] = size // record_size

            if 'record' in gp:
                del gp['record']
        else:
            stream.seek(0, 2)
            addr = stream.tell()
            gp['data_block'].append(addr)
            size = len(samples)
            stream.write(samples)

            record_size = gp['channel_group']['samples_byte_nr']
            record_size += gp['data_group']['record_id_len']
            added_cycles = size // record_size
            gp['channel_group']['cycles_nr'] += added_cycles

            gp['data_block_addr'].append(addr)
            gp['data_size'].append(size)
            gp['data_block_size'].append(size)

        del samples

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

        Returns
        -------
        index : int
            new attachment index

        """
        if data in self._attachments_cache:
            return self._attachments_cache[data]
        else:
            creator_index = len(self.file_history)
            fh = FileHistory()
            fh.comment = """<FHcomment>
<TX>Added new embedded attachment from {}</TX>
<tool_id>asammdf</tool_id>
<tool_vendor>asammdf</tool_vendor>
<tool_version>{}</tool_version>
</FHcomment>""".format(
                file_name if file_name else 'bin.bin',
                __version__,
            )

            self.file_history.append(fh)

            at_block = AttachmentBlock(data=data, compression=compression)
            at_block['creator_index'] = creator_index
            index = v4c.MAX_UINT64
            while index in self.attachments:
                index -= 1
            self.attachments[index] = at_block

            at_block.file_name = file_name if file_name else 'bin.bin'
            at_block.mime = mime
            at_block.comment = comment

            self._attachments_cache[data] = index
            return index

    def close(self):
        """ if the MDF was created with memory=False and new
        channels have been appended, then this must be called just before the
        object is not used anymore to clean-up the temporary file"""
        if self._tempfile is not None:
            self._tempfile.close()
        if self._file is not None:
            self._file.close()

    def extract_attachment(self, address=None, index=None):
        """ extract attachment data by original address or by index. If it is an embedded attachment,
        then this method creates the new file according to the attachment file
        name information

        Parameters
        ----------
        address : int
            attachment index; default *None*
        index : int
            attachment index; default *None*

        Returns
        -------
        data : bytes | str
            attachment data

        """
        if address is None and index is None:
            return b'', ''

        if address is not None:
            index = self._attachments_map[address]
        attachment = self.attachments[index]

        current_path = os.getcwd()
        file_path = attachment.file_name or 'embedded'
        try:
            os.chdir(os.path.dirname(self.name))

            flags = attachment['flags']

            # for embedded attachments extrat data and create new files
            if flags & v4c.FLAG_AT_EMBEDDED:
                data = attachment.extract()

                return data, file_path
            else:
                # for external attachments read the file and return the content
                if flags & v4c.FLAG_AT_MD5_VALID:
                    data = open(file_path, 'rb').read()
                    md5_worker = md5()
                    md5_worker.update(data)
                    md5_sum = md5_worker.digest()
                    if attachment['md5_sum'] == md5_sum:
                        if attachment.mime.startswith('text'):
                            with open(file_path, 'r') as f:
                                data = f.read()
                        return data, file_path
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
                        logger.warning(message)
                else:
                    if attachment.mime.startswith('text'):
                        mode = 'r'
                    else:
                        mode = 'rb'
                    with open(file_path, mode) as f:
                        data = f.read()
                    return data, file_path
        except Exception as err:
            os.chdir(current_path)
            message = 'Exception during attachment extraction: ' + repr(err)
            logger.warning(message)
            return b'', file_path

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

        if self.memory == 'minimum':

            channel = Channel(
                address=channel,
                stream=stream,
            )

        conversion = channel.conversion

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

        return extract_cncomment_xml(channel.comment)

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

        name = channel.name

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

        if grp['data_location'] == v4c.LOCATION_ORIGINAL_FILE:
            stream = self._file
        else:
            stream = self._tempfile

        if ch_nr >= 0:
            channel = grp['channels'][ch_nr]

            if self.memory == 'minimum':
                channel = Channel(
                    address=channel,
                    stream=stream,
                )
        else:
            channel = grp['logging_channels'][-ch_nr -1]

        return channel

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
            returns *Signal* if *samples_only* = *False* (default option),
            otherwise returns numpy.array
            The *Signal* samples are:

                * numpy recarray for channels that have composition/channel
                  array address or for channel of type
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

        if ch_nr >= 0:

            # get the channel object
            if memory == 'minimum':
                if samples_only and raw:
                    channel = Channel(
                        address=grp['channels'][ch_nr],
                        stream=stream,
                        load_metadata=False,
                    )
                else:
                    channel = Channel(
                        address=grp['channels'][ch_nr],
                        stream=stream,
                        cc_map=self._cc_map,
                        si_map=self._si_map,
                    )
            else:
                channel = grp['channels'][ch_nr]

            dependency_list = grp['channel_dependencies'][ch_nr]

            if data:
                cycles_nr = len(data[0]) // grp['channel_group']['samples_byte_nr']
            else:
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
            else:
                data = (data, )

            channel_invalidation_present = (
                channel['flags']
                & (v4c.FLAG_INVALIDATION_BIT_VALID | v4c.FLAG_ALL_SAMPLES_VALID)
                == v4c.FLAG_INVALIDATION_BIT_VALID
            )

            # get the channel signal data if available
            signal_data = self._load_signal_data(
                group=grp,
                index=ch_nr,
            )

            bit_count = channel['bit_count']
        else:
            # get data group record
            try:
                parents, dtypes = grp['parents'], grp['types']
            except KeyError:
                grp['parents'], grp['types'] = self._prepare_record(grp)
                parents, dtypes = grp['parents'], grp['types']
            if data:
                cycles_nr = len(data[0]) // grp['channel_group']['samples_byte_nr']
            else:
                cycles_nr = grp['channel_group']['cycles_nr']

            parent, bit_offset = parents[ch_nr]

            channel_invalidation_present = False
            dependency_list = None

            channel = grp['logging_channels'][-ch_nr-1]


            # get group data
            if data is None:
                data = self._load_group_data(grp)
            else:
                data = (data,)

            bit_count = channel['bit_count']

        data_type = channel['data_type']
        channel_type = channel['channel_type']

        # check if this is a channel array
        if dependency_list:
            arrays = []
            if name is None:
                name = channel.name

            if all(
                    not isinstance(dep, ChannelArrayBlock)
                    for dep in dependency_list):
                # structure channel composition

                if memory == 'minimum':
                    names = []

                    for ch_nr, _ in dependency_list:
                        address = grp['channels'][ch_nr]
                        channel = Channel(
                            address=address,
                            stream=stream,
                            load_metadata=False,
                        )

                        name_ = get_text_v4(
                            address=channel['name_addr'],
                            stream=stream,
                        )
                        names.append(name_)
                else:
                    names = [
                        grp['channels'][ch_nr].name
                        for ch_nr, _ in dependency_list
                    ]

                channel_values = [
                    []
                    for _ in dependency_list
                ]
                timestamps = []
                valid_indexes = []

                count = 0
                for fragment in data:
                    for i, (ch_nr, dg_nr) in enumerate(dependency_list):
                        vals = self.get(
                            group=dg_nr,
                            index=ch_nr,
                            samples_only=True,
                            raw=raw,
                            data=fragment,
                        )
                        channel_values[i].append(vals)
                    if not samples_only or raster:
                        timestamps.append(self.get_master(gp_nr, fragment))
                    if channel_invalidation_present:
                        valid_indexes.append(
                            self.get_valid_indexes(gp_nr, channel, fragment)
                        )

                    count += 1

                if count > 1:
                    arrays = [concatenate(lst) for lst in channel_values]
                else:
                    arrays = [lst[0] for lst in channel_values]
                types = [
                    (name_, arr.dtype, arr.shape[1:])
                    for name_, arr in zip(names, arrays)
                ]
                if PYVERSION == 2:
                    types = fix_dtype_fields(types)
                types = dtype(types)

                vals = fromarrays(arrays, dtype=types)

                if not samples_only or raster:
                    if count > 1:
                        timestamps = concatenate(timestamps)
                    else:
                        timestamps = timestamps[0]

                if channel_invalidation_present:
                    if count > 1:
                        valid_indexes = concatenate(valid_indexes)
                    else:
                        valid_indexes = valid_indexes[0]
                    vals = vals[valid_indexes]
                    if not samples_only or raster:
                        timestamps = timestamps[valid_indexes]

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

                cycles_nr = len(vals)

            else:
                # channel arrays

                channel_values = []
                timestamps = []
                valid_indexes = []
                count = 0
                for fragment in data:

                    data_bytes, offset = fragment

                    arrays = []
                    types = []
                    try:
                        parent, bit_offset = parents[ch_nr]
                    except KeyError:
                        parent, bit_offset = None, None

                    if parent is not None:
                        if 'record' not in grp:
                            if dtypes.itemsize:
                                record = fromstring(data_bytes, dtype=dtypes)
                            else:
                                record = None

                            if self.memory == 'full':
                                grp['record'] = record
                        else:
                            record = grp['record']

                        record.setflags(write=False)

                        vals = record[parent]
                    else:
                        vals = self._get_not_byte_aligned_data(
                            data_bytes,
                            grp,
                            ch_nr,
                        )

                    vals = vals.copy()

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
                                    axis = array(
                                        [axis for _ in range(cycles_nr)]
                                    )
                                    arrays.append(axis)
                                    dtype_pair = (
                                        'axis_{}'.format(i),
                                        axis.dtype,
                                        shape,
                                    )
                                    types.append(dtype_pair)
                            else:
                                for i in range(dims_nr):
                                    ref_ch_nr, ref_dg_nr = ca_block.referenced_channels[i]
                                    if memory == 'minimum':
                                        address = (
                                            self.groups[ref_dg_nr]
                                            ['channels']
                                            [ref_ch_nr]
                                        )
                                        ref_channel = Channel(
                                            address=address,
                                            stream=stream,
                                            cc_map=self._cc_map,
                                            si_map=self._si_map,
                                        )
                                        axisname = ref_channel.name
                                    else:
                                        axisname = (
                                            self.groups[ref_dg_nr]
                                            ['channels']
                                            [ref_ch_nr]
                                            .name
                                        )

                                    shape = (ca_block['dim_size_{}'.format(i)],)
                                    if ref_dg_nr == gp_nr:
                                        axis_values = self.get(
                                            group=ref_dg_nr,
                                            index=ref_ch_nr,
                                            samples_only=True,
                                            data=fragment,
                                        )
                                    else:
                                        channel_group = grp['channel_group']
                                        record_size = channel_group['samples_byte_nr']
                                        record_size += channel_group['invalidation_bytes_nr']
                                        start = offset // record_size
                                        end = start + len(data_bytes) // record_size + 1
                                        ref = self.get(
                                            group=ref_dg_nr,
                                            index=ref_ch_nr,
                                            samples_only=True,
                                        )
                                        axis_values = ref[start: end].copy()
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
                                types.append(
                                    ('axis_{}'.format(i), axis.dtype, shape)
                                )
                        else:
                            for i in range(dims_nr):
                                ref_ch_nr, ref_dg_nr = ca_block.referenced_channels[i]
                                if memory == 'minimum':
                                    address = (
                                        self.groups[ref_dg_nr]
                                        ['channels']
                                        [ref_ch_nr]
                                    )
                                    ref_channel = Channel(
                                        address=address,
                                        stream=stream,
                                        cc_map=self._cc_map,
                                        si_map=self._si_map,
                                    )
                                    axisname = ref_channel.name
                                else:
                                    axisname = (
                                        self.groups[ref_dg_nr]
                                        ['channels']
                                        [ref_ch_nr]
                                        .name
                                    )

                                shape = (ca_block['dim_size_{}'.format(i)],)
                                if ref_dg_nr == gp_nr:
                                    axis_values = self.get(
                                        group=ref_dg_nr,
                                        index=ref_ch_nr,
                                        samples_only=True,
                                        data=fragment,
                                    )
                                else:
                                    channel_group = grp['channel_group']
                                    record_size = channel_group['samples_byte_nr']
                                    record_size += channel_group['invalidation_bytes_nr']
                                    start = offset // record_size
                                    end = start + len(data_bytes) // record_size + 1
                                    ref = self.get(
                                        group=ref_dg_nr,
                                        index=ref_ch_nr,
                                        samples_only=True,
                                    )
                                    axis_values = ref[start: end].copy()
                                axis_values = axis_values[axisname]

                                arrays.append(axis_values)
                                dtype_pair = axisname, axis_values.dtype, shape
                                types.append(dtype_pair)

                    if PYVERSION == 2:
                        types = fix_dtype_fields(types)

                    vals = fromarrays(arrays, dtype(types))

                    if not samples_only or raster:
                        timestamps.append(self.get_master(gp_nr, fragment))
                    if channel_invalidation_present:
                        valid_indexes.append(
                            self.get_valid_indexes(gp_nr, channel, fragment)
                        )

                    channel_values.append(vals)
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

                if channel_invalidation_present:
                    if count > 1:
                        valid_indexes = concatenate(valid_indexes)
                    else:
                        valid_indexes = valid_indexes[0]
                    vals = vals[valid_indexes]
                    if not samples_only or raster:
                        timestamps = timestamps[valid_indexes]

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

                cycles_nr = len(vals)

            conversion = channel.conversion

        else:
            # get channel values
            if channel['channel_type'] in (v4c.CHANNEL_TYPE_VIRTUAL,
                                           v4c.CHANNEL_TYPE_VIRTUAL_MASTER):
                data_type = channel['data_type']
                ch_dtype = dtype(get_fmt_v4(data_type, 64))

                channel_values = []
                timestamps = []
                valid_indexes = []

                channel_group = grp['channel_group']
                record_size = channel_group['samples_byte_nr']
                record_size += channel_group['invalidation_bytes_nr']

                count = 0
                for fragment in data:
                    data_bytes, offset = fragment
                    offset = offset // record_size

                    vals = arange(len(data_bytes)//record_size, dtype=ch_dtype)
                    vals += offset

                    if not samples_only or raster:
                        timestamps.append(self.get_master(gp_nr, fragment))
                    if channel_invalidation_present:
                        valid_indexes.append(
                            self.get_valid_indexes(gp_nr, channel, fragment)
                        )

                    channel_values.append(vals)
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

                if channel_invalidation_present:
                    if count > 1:
                        valid_indexes = concatenate(valid_indexes)
                    else:
                        valid_indexes = valid_indexes[0]
                    vals = vals[valid_indexes]
                    if not samples_only or raster:
                        timestamps = timestamps[valid_indexes]

                if raster:
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
                channel_values = []
                timestamps = []
                valid_indexes = []

                count = 0
                for fragment in data:

                    data_bytes, offset = fragment
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

                        size = vals.dtype.itemsize
                        for dim in vals.shape[1:]:
                            size *= dim
                        data_type = channel['data_type']

                        vals_dtype = vals.dtype.kind

                        if vals_dtype == 'b':
                            pass
                        elif vals_dtype not in 'ui' and (bit_offset or not bits == size * 8) or \
                                (len(vals.shape) > 1 and data_type != v4c.DATA_TYPE_BYTEARRAY):
                            vals = self._get_not_byte_aligned_data(
                                data_bytes,
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
                                if data_type in v4c.SIGNED_INT:
                                    vals = as_non_byte_sized_signed_int(
                                        vals,
                                        bits,
                                    )
                                else:
                                    mask = (1 << bits) - 1
                                    if vals.flags.writeable:
                                        vals &= mask
                                    else:
                                        vals = vals & mask
                    else:
                        vals = self._get_not_byte_aligned_data(
                            data_bytes,
                            grp,
                            ch_nr,
                        )

                    if bits == 1 and self._single_bit_uint_as_bool:
                        vals = array(vals, dtype=bool)
                    else:
                        data_type = channel['data_type']
                        channel_dtype = array(
                            [],
                            dtype=get_fmt_v4(
                                data_type,
                                bits,
                                channel_type,
                            ),
                        )
                        if vals.dtype != channel_dtype.dtype:
                            vals = vals.astype(channel_dtype.dtype)

                    if not samples_only or raster:
                        timestamps.append(self.get_master(gp_nr, fragment))
                    if channel_invalidation_present:
                        valid_indexes.append(
                            self.get_valid_indexes(gp_nr, channel, fragment)
                        )
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

                if channel_invalidation_present:
                    if count > 1:
                        valid_indexes = concatenate(valid_indexes)
                    else:
                        valid_indexes = valid_indexes[0]
                    vals = vals[valid_indexes]
                    if not samples_only or raster:
                        timestamps = timestamps[valid_indexes]

                if raster:
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


            # get the channel conversion
            conversion = channel.conversion

            if conversion is None:
                conversion_type = v4c.CONVERSION_TYPE_NON
            else:
                conversion_type = conversion['conversion_type']

            if conversion_type in (
                    v4c.CONVERSION_TYPE_NON,
                    v4c.CONVERSION_TYPE_TRANS,
                    v4c.CONVERSION_TYPE_TTAB):

                if channel_type == v4c.CHANNEL_TYPE_VLSD:
                    if signal_data:
                        values = []
                        for offset in vals:
                            offset = int(offset)
                            str_size = unpack_from('<I', signal_data, offset)[0]
                            values.append(
                                signal_data[offset + 4: offset + 4 + str_size]
                            )

                        if data_type == v4c.DATA_TYPE_BYTEARRAY:

                            if PYVERSION >= 3:
                                values = [
                                    list(val)
                                    for val in values
                                ]
                            else:
                                values = [
                                    [ord(byte) for byte in val]
                                    for val in values
                                ]

                            dim = max(len(arr) for arr in values) if values else 0

                            for lst in values:
                                lst.extend([0, ] * (dim - len(lst)))

                            vals = array(
                                values,
                                dtype=uint8,
                            )

                        else:

                            vals = array(values)

                            if data_type == v4c.DATA_TYPE_STRING_UTF_16_BE:
                                encoding = 'utf-16-be'

                            elif data_type == v4c.DATA_TYPE_STRING_UTF_16_LE:
                                encoding = 'utf-16-le'

                            elif data_type == v4c.DATA_TYPE_STRING_UTF_8:
                                encoding = 'utf-8'

                            elif data_type == v4c.DATA_TYPE_STRING_LATIN_1:
                                encoding = 'latin-1'

                            if encoding != 'latin-1':

                                if encoding == 'utf-16-le':
                                    vals = vals.view(uint16).byteswap().view(vals.dtype)
                                    vals = encode(decode(vals, 'utf-16-be'), 'latin-1')
                                else:
                                    vals = encode(decode(vals, encoding), 'latin-1')
                    else:
                        # no VLSD signal data samples
                        vals = array([], dtype=dtype('S'))

                elif channel_type in (v4c.CHANNEL_TYPE_VALUE, v4c.CHANNEL_TYPE_MLSD) and \
                    (v4c.DATA_TYPE_STRING_LATIN_1 <= data_type <= v4c.DATA_TYPE_STRING_UTF_16_BE):

                    if data_type == v4c.DATA_TYPE_STRING_UTF_16_BE:
                        encoding = 'utf-16-be'

                    elif data_type == v4c.DATA_TYPE_STRING_UTF_16_LE:
                        encoding = 'utf-16-le'

                    elif data_type == v4c.DATA_TYPE_STRING_UTF_8:
                        encoding = 'utf-8'

                    elif data_type == v4c.DATA_TYPE_STRING_LATIN_1:
                        encoding = 'latin-1'

                    if encoding != 'latin-1':
                        if encoding == 'utf-16-le':
                            vals = vals.view(uint16).byteswap().view(vals.dtype)
                            vals = encode(decode(vals, 'utf-16-be'), 'latin-1')
                        else:
                            vals = encode(decode(vals, encoding), 'latin-1')

                # CANopen date
                if data_type == v4c.DATA_TYPE_CANOPEN_DATE:

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

                if conversion_type == v4c.CONVERSION_TYPE_TRANS:
                    if not raw:
                        vals = conversion.convert(vals)
                if conversion_type == v4c.CONVERSION_TYPE_TTAB:
                    raw = True

            elif conversion_type in (
                    v4c.CONVERSION_TYPE_LIN,
                    v4c.CONVERSION_TYPE_RAT,
                    v4c.CONVERSION_TYPE_ALG,
                    v4c.CONVERSION_TYPE_TABI,
                    v4c.CONVERSION_TYPE_TAB,
                    v4c.CONVERSION_TYPE_RTAB):
                if not raw:
                    vals = conversion.convert(vals)

            elif conversion_type in (
                    v4c.CONVERSION_TYPE_TABX,
                    v4c.CONVERSION_TYPE_RTABX):
                raw = True

        if samples_only:
            res = vals
        else:
            # search for unit in conversion texts

            if name is None:
                name = channel.name

            unit = (
                conversion and conversion.unit
                or channel.unit
            )

            if unit:
                unit = unit.strip(' \t\r\n\0')

            comment = channel.comment

            source = channel.source
            cg_source = grp['channel_group'].acq_source
            if source:
                source = SignalSource(
                    source.name or (cg_source and cg_source.name) or '',
                    source.path,
                    source.comment,
                    source['source_type'],
                    source['bus_type'],
                )
            else:
                source = None

            if channel.attachments:
                attachment = self.extract_attachment(index=channel.attachments[0])
            else:
                attachment = ()

            master_metadata = self._master_channel_metadata.get(gp_nr, None)

            res = Signal(
                samples=vals,
                timestamps=timestamps,
                unit=unit,
                name=name,
                comment=comment,
                conversion=conversion,
                raw=raw,
                master_metadata=master_metadata,
                attachment=attachment,
                source=source,
                display_name=channel.display_name,
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
                offset = 0

        group = self.groups[index]

        original_data = fragment

        if group['data_location'] == v4c.LOCATION_ORIGINAL_FILE:
            stream = self._file
        else:
            stream = self._tempfile
        memory = self.memory

        time_ch_nr = self.masters_db.get(index, None)
        channel_group = group['channel_group']
        record_size = channel_group['samples_byte_nr']
        record_size += channel_group['invalidation_bytes_nr']
        cycles_nr = group['channel_group']['cycles_nr']

        if original_data:
            cycles_nr = len(data_bytes) // record_size

        if time_ch_nr is None:
            offset = offset // record_size
            t = arange(cycles_nr, dtype=float64)
            t += offset
            metadata = (
                'Time',
                v4c.SYNC_TYPE_TIME,
            )
        else:

            time_ch = group['channels'][time_ch_nr]
            if memory == 'minimum':
                time_ch = Channel(
                    address=group['channels'][time_ch_nr],
                    stream=stream,
                    cc_map=self._cc_map,
                    si_map=self._si_map,
                )
            time_conv = time_ch.conversion
            time_name = time_ch.name

            metadata = (
                time_name,
                time_ch['sync_type'],
            )

            if time_ch['channel_type'] == v4c.CHANNEL_TYPE_VIRTUAL_MASTER:
                offset = offset // record_size
                time_a = time_conv['a']
                time_b = time_conv['b']
                t = arange(cycles_nr, dtype=float64)
                t += offset
                t *= time_a
                t += time_b

            else:
                # get data group parents and dtypes
                try:
                    parents, dtypes = group['parents'], group['types']
                except KeyError:
                    parents, dtypes = self._prepare_record(group)
                    group['parents'], group['types'] = parents, dtypes

                # get data
                if fragment is None:
                    data = self._load_group_data(group)
                else:
                    data = (fragment, )
                time_values = []

                for fragment in data:
                    data_bytes, offset = fragment
                    try:
                        parent, _ = parents[time_ch_nr]
                    except KeyError:
                        parent = None
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
                            data_bytes, group,
                            time_ch_nr,
                        )

                    time_values.append(t.copy())

                if len(time_values) > 1:
                    t = concatenate(time_values)
                else:
                    t = time_values[0]

                # get timestamps
                if time_conv:
                    t = time_conv.convert(t)

        self._master_channel_metadata[index] = metadata

        if not t.dtype == float64:
            t = t.astype(float64)

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
                name = channel.name

                ch_type = v4c.CHANNEL_TYPE_TO_DESCRIPTION[channel['channel_type']]
                inf['channel {}'.format(j)] = 'name="{}" type={}'.format(
                    name,
                    ch_type,
                )

        return info

    def save(self, dst='', overwrite=False, compression=0):
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

        if self.name is None and dst == '':
            message = (
                'Must specify a destination file name '
                'for MDF created from scratch'
            )
            raise MdfException(message)

        _read_fragment_size = self._read_fragment_size
        self.configure(read_fragment_size=4 * 2 ** 20)

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
                logger.warning(message)
                dst = name

        if not self.file_history:
            comment = 'created'
        else:
            comment = 'updated'

        fh = FileHistory()
        fh.comment = """<FHcomment>
<TX>{}</TX>
<tool_id>asammdf</tool_id>
<tool_vendor>asammdf</tool_vendor>
<tool_version>{}</tool_version>
</FHcomment>""".format(comment, __version__)

        self.file_history.append(fh)

        if self.memory == 'low' and dst == self.name:
            destination = dst + '.temp'
        else:
            destination = dst

        with open(destination, 'wb+') as dst_:
            defined_texts = {}
            cc_map = {}
            si_map = {}

            groups_nr = len(self.groups)

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
            for gp_nr, gp in enumerate(self.groups):
                original_data_addresses.append(
                    gp['data_group']['data_block_addr']
                )

                if gp['channel_group']['flags'] & v4c.FLAG_CG_VLSD:
                    continue

                address = tell()

                data = self._load_group_data(gp)

                total_size = gp['channel_group']['samples_byte_nr'] * gp['channel_group']['cycles_nr']

                if self._write_fragment_size:

                    samples_size = gp['channel_group']['samples_byte_nr']
                    split_size = self._write_fragment_size // samples_size
                    split_size *= samples_size
                    if split_size == 0:
                        chunks = 1
                    else:
                        chunks = float(total_size) / split_size
                        chunks = int(ceil(chunks))
                else:
                    chunks = 1

                if chunks == 1:
                    if PYVERSION == 3:
                        data = b''.join(d[0] for d in data)
                    else:
                        data = b''.join(str(d[0]) for d in data)
                    if compression and self.version > '4.00':
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

                    cur_data = b''

                    if self.memory == 'low':
                        for i in range(chunks):
                            while len(cur_data) < split_size:
                                try:
                                    cur_data += next(data)[0]
                                except StopIteration:
                                    break

                            data_, cur_data = cur_data[:split_size], cur_data[split_size:]
                            if compression and self.version > '4.00':
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

                            write(bytes(block))

                            align = block['block_len'] % 8
                            if align:
                                write(b'\0' * (8 - align))
                            dl_block['data_block_addr{}'.format(i)] = address
                    else:
                        cur_data = next(data)[0]
                        for i in range(chunks):

                            data_ = cur_data[i*split_size: (i + 1) * split_size]
                            if compression and self.version > '4.00':
                                if compression == 1:
                                    zip_type = v4c.FLAG_DZ_DEFLATE
                                    param = 0
                                else:
                                    zip_type = v4c.FLAG_DZ_TRANPOSED_DEFLATE
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

                if self._callback:
                    self._callback(int(50 * (gp_nr+1) / groups_nr), 100)
                if self._terminate:
                    dst_.close()
                    self.close()
                    return

            address = tell()

            blocks = []

            if self.header.comment:
                meta = self.header.comment.startswith('<HDcomment')
                block = TextBlock(
                    text=self.header.comment,
                    meta=meta,
                )
                blocks.append(block)
                self.header['comment_addr'] = address
                address += block['block_len']

            # attachments
            at_map = {}
            if self.attachments:
                for at_block in self.attachments:
                    address = at_block.to_blocks(address, blocks, defined_texts)

                for i in range(len(self.attachments) - 1):
                    at_block = self.attachments[i]
                    at_block['next_at_addr'] = self.attachments[i+1].address
                self.attachments[-1]['next_at_addr'] = 0

            # file history blocks
            for fh in self.file_history:
                address = fh.to_blocks(address, blocks, defined_texts)

            for i, fh in enumerate(self.file_history[:-1]):
                fh['next_fh_addr'] = self.file_history[i + 1].address
            self.file_history[-1]['next_fh_addr'] = 0

            # data groups
            gp_rec_ids = []
            valid_data_groups = []
            for gp in self.groups:
                if gp['channel_group']['flags'] & v4c.FLAG_CG_VLSD:
                    continue

                valid_data_groups.append(gp['data_group'])
                gp_rec_ids.append(gp['data_group']['record_id_len'])

                address = gp['data_group'].to_blocks(address, blocks, defined_texts)

            if valid_data_groups:
                for i, dg in enumerate(valid_data_groups[:-1]):
                    addr_ = valid_data_groups[i + 1].address
                    dg['next_dg_addr'] = addr_
                valid_data_groups[-1]['next_dg_addr'] = 0

            # go through each data group and append the rest of the blocks
            for i, gp in enumerate(self.groups):

                for channel in gp['channels']:
                    for j, idx in enumerate(channel.attachments):
                        key = 'attachment_{}_addr'.format(j)
                        channel[key] = self.attachments[idx].address

                    address = channel.to_blocks(address, blocks, defined_texts, cc_map, si_map)

                # channel data
                gp_sd = []
                for j, sdata in enumerate(gp['signal_data']):
                    sdata = self._load_signal_data(
                        group=gp,
                        index=j,
                    )
                    if sdata:
                        if compression and self.version > '4.00':
                            signal_data = DataZippedBlock(
                                data=sdata,
                                zip_type=v4c.FLAG_DZ_DEFLATE,
                                original_type=b'SD',
                            )
                            signal_data.address = address
                            address += signal_data['block_len']
                            blocks.append(signal_data)
                            align = signal_data['block_len'] % 8
                            if align % 8:
                                blocks.append(b'\0' * (8 - align))
                                address += 8 - align
                        else:
                            signal_data = SignalDataBlock(data=sdata)
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
                        zip(gp['channels'], gp_sd)):

                    if signal_data:
                        channel['data_block_addr'] = signal_data.address
                    else:
                        channel['data_block_addr'] = 0

                    if gp['channel_dependencies'][j]:
                        dep = gp['channel_dependencies'][j][0]
                        if isinstance(dep, tuple):
                            index = dep[0]
                            addr_ = gp['channels'][index].address
                        else:
                            addr_ = dep.address
                        channel['component_addr'] = addr_

                for channel in gp['logging_channels']:
                    address = channel.to_blocks(address, blocks, defined_texts, cc_map, si_map)

                group_channels = list(chain(gp['channels'], gp['logging_channels']))
                if group_channels:
                    for j, channel in enumerate(group_channels[:-1]):
                        channel['next_ch_addr'] = group_channels[j + 1].address
                    group_channels[-1]['next_ch_addr'] = 0

                # channel dependecies
                j = len(gp['channels']) - 1
                while j >= 0:
                    dep_list = gp['channel_dependencies'][j]
                    if dep_list and all(
                            isinstance(dep, tuple) for dep in dep_list):
                        index = dep_list[0][0]
                        gp['channels'][j]['component_addr'] = gp['channels'][index].address
                        index = dep_list[-1][0]
                        gp['channels'][j]['next_ch_addr'] = gp['channels'][index]['next_ch_addr']
                        gp['channels'][index]['next_ch_addr'] = 0

                        for ch_nr, _ in dep_list:
                            gp['channels'][ch_nr]['source_addr'] = 0
                    j -= 1

                # channel group
                if gp['channel_group']['flags'] & v4c.FLAG_CG_VLSD:
                    continue

                if gp['channels']:
                    gp['channel_group']['first_ch_addr'] = gp['channels'][0].address
                else:
                    gp['channel_group']['first_ch_addr'] = 0
                gp['channel_group']['next_cg_addr'] = 0

                address = gp['channel_group'].to_blocks(address, blocks, defined_texts, si_map)
                gp['data_group']['first_cg_addr'] = gp['channel_group'].address

                if self._callback:
                    self._callback(int(50 * (i+1) / groups_nr) + 25, 100)
                if self._terminate:
                    dst_.close()
                    self.close()
                    return

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

            for gp in self.groups:
                gp['data_group']['record_id_len'] = 0

            ev_map = []

            if self.events:
                for event in self.events:
                    for i, ref in enumerate(event.scopes):
                        try:
                            ch_cntr, dg_cntr = ref
                            event['scope_{}_addr'.format(i)] = (
                                self.groups
                                [dg_cntr]
                                ['channels']
                                [ch_cntr]
                                .address
                            )
                        except TypeError:
                            dg_cntr = ref
                            event['scope_{}_addr'.format(i)] = (
                                self.groups
                                [dg_cntr]
                                ['channel_group']
                                .address
                            )
                    for i in range(event['attachment_nr']):
                        key = 'attachment_{}_addr'.format(i)
                        addr = event[key]
                        event[key] = at_map[addr]

                    blocks.append(event)
                    ev_map.append(address)
                    event.address = address
                    address += event['block_len']

                    if event.name:
                        tx_block = TextBlock(text=event.name)
                        tx_block.address = address
                        blocks.append(tx_block)
                        address += tx_block['block_len']
                        event['name_addr'] = tx_block.address
                    else:
                        event['name_addr'] = 0

                    if event.comment:
                        meta = event.comment.startswith('<EVcomment')
                        tx_block = TextBlock(text=event.comment, meta=meta)
                        tx_block.address = address
                        blocks.append(tx_block)
                        address += tx_block['block_len']
                        event['comment_addr'] = tx_block.address
                    else:
                        event['comment_addr'] = 0

                    if event.parent is not None:
                        event['parent_ev_addr'] = ev_map[event.parent]
                    if event.range_start is not None:
                        event['range_start_ev_addr'] = ev_map[event.range_start]

                for i in range(len(self.events) - 1):
                    self.events[i]['next_ev_addr'] = self.events[i+1].address
                self.events[-1]['next_ev_addr'] = 0

                self.header['first_event_addr'] = self.events[0].address

            if self._terminate:
                dst_.close()
                self.close()
                return

            if self._callback:
                blocks_nr = len(blocks)
                threshold = blocks_nr / 25
                count = 1
                for i, block in enumerate(blocks):
                    write(bytes(block))
                    if i >= threshold:
                        self._callback(75 + count, 100)
                        count += 1
                        threshold += blocks_nr / 25
            else:
                for block in blocks:
                    write(bytes(block))

            for gp, rec_id in zip(self.groups, gp_rec_ids):
                gp['data_group']['record_id_len'] = rec_id

            if valid_data_groups:
                addr_ = valid_data_groups[0].address
                self.header['first_dg_addr'] = addr_
            else:
                self.header['first_dg_addr'] = 0
            self.header['file_history_addr'] = self.file_history[0].address
            if self.attachments:
                first_attachment = self.attachments[0]
                addr_ = first_attachment.address
                self.header['first_attachment_addr'] = addr_
            else:
                self.header['first_attachment_addr'] = 0

            seek(v4c.IDENTIFICATION_BLOCK_SIZE)
            write(bytes(self.header))

            for orig_addr, gp in zip(original_data_addresses, self.groups):
                gp['data_group']['data_block_addr'] = orig_addr

            at_map = {value:key for key, value in at_map.items()}

            for event in self.events:
                for i in range(event['attachment_nr']):
                    key = 'attachment_{}_addr'.format(i)
                    addr = event[key]
                    event[key] = at_map[addr]

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
                logger.warning(message)
                dst = name

        if not self.file_history:
            comment = 'created'
        else:
            comment = 'updated'

        fh = FileHistory()
        fh.comment = """<FHcomment>
<TX>{}</TX>
<tool_id>asammdf</tool_id>
<tool_vendor>asammdf</tool_vendor>
<tool_version>{}</tool_version>
</FHcomment>""".format(comment, __version__)

        self.file_history.append(fh)

        if dst == self.name:
            destination = dst + '.temp'
        else:
            destination = dst

        with open(destination, 'wb+') as dst_:
            defined_texts = {}
            cc_map = {}
            si_map = {}

            groups_nr = len(self.groups)

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
            for group_index, gp in enumerate(self.groups):
                original_data_addresses.append(
                    gp['data_group']['data_block_addr']
                )

                if gp['channel_group']['flags'] & v4c.FLAG_CG_VLSD:
                    continue

                address = tell()

                data = self._load_group_data(gp)

                if self._write_fragment_size:
                    total_size = gp['channel_group']['samples_byte_nr'] * gp['channel_group']['cycles_nr']
                    samples_size = gp['channel_group']['samples_byte_nr']
                    split_size = self._write_fragment_size // samples_size
                    split_size *= samples_size
                    if split_size == 0:
                        chunks = 1
                    else:
                        chunks = total_size / split_size
                        chunks = int(ceil(chunks))
                else:
                    chunks = 1

                if chunks == 1:
                    data = b''.join(d[0] for d in data)
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

                    cur_data = b''

                    for i in range(chunks):
                        while len(cur_data) < split_size:
                            try:
                                cur_data += next(data)[0]
                            except StopIteration:
                                break

                        data_, cur_data = cur_data[:split_size], cur_data[split_size:]
                        if compression and self.version > '4.00':
                            if compression == 1:
                                zip_type = v4c.FLAG_DZ_DEFLATE
                                param = 0
                            else:
                                zip_type = v4c.FLAG_DZ_TRANPOSED_DEFLATE
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

                if self._callback:
                    self._callback(int(50 * (group_index+1) / groups_nr), 100)
                if self._terminate:
                    dst_.close()
                    self.close()
                    return

            address = tell()

            if self.header.comment:
                meta = self.header.comment.startswith('<HDcomment')
                block = TextBlock(
                    text=self.header.comment,
                    meta=meta,
                )
                write(bytes(block))
                self.header['comment_addr'] = address
            else:
                self.header['comment_addr'] = 0

            # attachments
            address = tell()
            blocks = []
            at_map = {}

            if self.attachments:
                for at_block in self.attachments:
                    address = at_block.to_blocks(address, blocks, defined_texts)

                for i in range(len(self.attachments) - 1):
                    at_block = self.attachments[i]
                    at_block['next_at_addr'] = self.attachments[i + 1].address
                self.attachments[-1]['next_at_addr'] = 0

            # file history blocks
            for fh in self.file_history:
                address = fh.to_blocks(address, blocks, defined_texts)

            for i, fh in enumerate(self.file_history[:-1]):
                fh['next_fh_addr'] = self.file_history[i + 1].address
            self.file_history[-1]['next_fh_addr'] = 0

            for blk in blocks:
                write(bytes(blk))

            del blocks

            address = tell()

            # go through each data group and append the rest of the blocks
            for i, gp in enumerate(self.groups):

                gp['temp_channels'] = ch_addrs = []

                if gp['data_location'] == v4c.LOCATION_ORIGINAL_FILE:
                    stream = self._file
                else:
                    stream = self._tempfile

                chans = gp['channels'] + gp['logging_channels']

                # channel dependecies
                structs = [
                    0 for _ in chans
                ]

                temp_deps = []
                incs = [0 for _ in chans]
                level = 0
                for j, dep_list in enumerate(gp['channel_dependencies'] + [None for _ in gp['logging_channels']]):
                    incs_ = [e for e in incs if e]
                    incs[level] -= 1
                    if incs[level] < 0:
                        incs[level] = 0
                    elif incs[level] == 0:
                        level -= 1
                    structs[j] = len(incs_)
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
                            level += 1
                            incs[level] = len(dep_list)
                            temp_deps.append([])
                            for _ in dep_list:
                                temp_deps[-1].append(0)
                    else:
                        temp_deps.append(0)

                next_ch_addr = [
                    0 for _ in range(max(structs) + 1)
                ]

                # channels
                address = blocks_start_addr = tell()

                size = len(chans)
                previous_level = structs[-1] if structs else 0
                for j in range(size-1, -1, -1):
                    channel = chans[j]
                    level = structs[j]

                    if not isinstance(channel, Channel):
                        channel = Channel(
                            address=channel,
                            stream=stream,
                            parse_xml_comment=False,
                        )

                    channel['next_ch_addr'] = next_ch_addr[level]
                    if level:
                        channel.source = None
                    elif temp_deps[j]:
                        channel['component_addr'] = temp_deps[j][0]
                    if level < previous_level:
                        channel['component_addr'] = next_ch_addr[previous_level]
                        next_ch_addr[previous_level] = 0

                    previous_level = level

                    try:
                        signal_data = self._load_signal_data(
                            group=gp,
                            index=j,
                        )
                    except IndexError:
                        signal_data = b''
                    if signal_data:
                        if compression and self.version > '4.00':
                            signal_data = DataZippedBlock(
                                data=signal_data,
                                zip_type=v4c.FLAG_DZ_DEFLATE,
                                original_type=b'SD',
                            )
                            channel['data_block_addr'] = address
                            address += signal_data['block_len']
                            write(bytes(signal_data))
                            align = signal_data['block_len'] % 8
                            if align % 8:
                                write(b'\0' * (8 - align))
                                address += 8 - align
                        else:
                            signal_data = SignalDataBlock(data=signal_data)
                            channel['data_block_addr'] = address
                            write(bytes(signal_data))
                            address += signal_data['block_len']
                            align = signal_data['block_len'] % 8
                            if align % 8:
                                write(b'\0' * (8 - align))
                                address += 8 - align
                    else:
                        channel['data_block_addr'] = 0

                    del signal_data

                    for j, idx in enumerate(channel.attachments):
                        key = 'attachment_{}_addr'.format(j)
                        channel[key] = self.attachments[idx].address

                    address = channel.to_stream(dst_, defined_texts, cc_map, si_map)
                    ch_addrs.append(channel.address)
                    next_ch_addr[level] = channel.address

                ch_addrs.reverse()

                gp['channel_group']['first_ch_addr'] = next_ch_addr[0]

                if gp['channel_group']['flags'] & v4c.FLAG_CG_VLSD:
                    continue

                # channel group
                gp['channel_group']['next_cg_addr'] = 0

                gp['channel_group'].to_stream(dst_, defined_texts, si_map)
                gp['data_group']['first_cg_addr'] = gp['channel_group'].address

                if self._callback:
                    self._callback(int(50 * (i+1) / groups_nr) + 50, 100)
                if self._terminate:
                    dst_.close()
                    self.close()
                    return

            blocks = []
            address = tell()
            gp_rec_ids = []
            valid_data_groups = []
            # data groups
            for gp in self.groups:

                gp_rec_ids.append(gp['data_group']['record_id_len'])
                if gp['channel_group']['flags'] & v4c.FLAG_CG_VLSD:
                    continue
                else:
                    valid_data_groups.append(gp['data_group'])

                    address = gp['data_group'].to_blocks(address, blocks, defined_texts)

            if valid_data_groups:
                for i, dg in enumerate(valid_data_groups[:-1]):
                    addr_ = valid_data_groups[i + 1].address
                    dg['next_dg_addr'] = addr_
                valid_data_groups[-1]['next_dg_addr'] = 0

            for gp in self.groups:
                gp['data_group']['record_id_len'] = 0

            ev_map = {}
            if self.events:
                for event in self.events:
                    for i, ref in enumerate(event.scopes):
                        try:
                            ch_cntr, dg_cntr = ref
                            event['scope_{}_addr'.format(i)] = (
                                self.groups
                                [dg_cntr]
                                ['channels']
                                [ch_cntr]
                                .address
                            )
                        except TypeError:
                            dg_cntr = ref
                            event['scope_{}_addr'.format(i)] = (
                                self.groups
                                [dg_cntr]
                                ['channel_group']
                                .address
                            )
                    for i in range(event['attachment_nr']):
                        key = 'attachment_{}_addr'.format(i)
                        addr = event[key]
                        event[key] = at_map[addr]

                    blocks.append(event)
                    ev_map[event.address] = address
                    event.address = address
                    address += event['block_len']

                    if event.name:
                        tx_block = TextBlock(text=event.name)
                        tx_block.address = address
                        blocks.append(tx_block)
                        address += tx_block['block_len']
                        event['name_addr'] = tx_block.address
                    else:
                        event['name_addr'] = 0

                    if event.comment:
                        meta = event.comment.startswith('<EVcomment')
                        tx_block = TextBlock(text=event.comment, meta=meta)
                        tx_block.address = address
                        blocks.append(tx_block)
                        address += tx_block['block_len']
                        event['comment_addr'] = tx_block.address
                    else:
                        event['comment_addr'] = 0

                for event in self.events:
                    if event['parent_ev_addr']:
                        event['parent_ev_addr'] = ev_map[event['parent_ev_addr']]
                    if event['range_start_ev_addr']:
                        event['range_start_ev_addr'] = ev_map[event['range_start_ev_addr']]

                for i in range(len(self.events) - 1):
                    self.events[i]['next_ev_addr'] = self.events[i+1].address
                self.events[-1]['next_ev_addr'] = 0

                self.header['first_event_addr'] = self.events[0].address

            for block in blocks:
                write(bytes(block))

            del blocks

            for gp, rec_id in zip(self.groups, gp_rec_ids):
                gp['data_group']['record_id_len'] = rec_id

            if valid_data_groups:
                addr_ = valid_data_groups[0].address
                self.header['first_dg_addr'] = addr_
            else:
                self.header['first_dg_addr'] = 0
            self.header['file_history_addr'] = self.file_history[0].address

            if self.attachments:
                first_attachment = self.attachments[0]
                addr_ = first_attachment.address
                self.header['first_attachment_addr'] = addr_
            else:
                self.header['first_attachment_addr'] = 0

            seek(v4c.IDENTIFICATION_BLOCK_SIZE)
            write(bytes(self.header))

            for orig_addr, gp in zip(original_data_addresses, self.groups):
                gp['data_group']['data_block_addr'] = orig_addr

            ev_map = {value: key for key, value in ev_map.items()}

            for event in self.events:
                if event['parent_ev_addr']:
                    event['parent_ev_addr'] = ev_map[event['parent_ev_addr']]
                if event['range_start_ev_addr']:
                    event['range_start_ev_addr'] = ev_map[event['range_start_ev_addr']]
                for i in range(event['attachment_nr']):
                    key = 'attachment_{}_addr'.format(i)
                    addr = event[key]
                    event[key] = at_map[addr]

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
                                seek(dep.address)
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


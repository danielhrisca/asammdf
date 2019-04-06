# -*- coding: utf-8 -*-
"""
classes that implement the blocks for MDF version 4
"""

import logging
import xml.etree.ElementTree as ET
import time
from datetime import datetime
from hashlib import md5
from struct import pack, unpack, unpack_from
from textwrap import wrap
from zlib import compress, decompress
from pathlib import Path

import numpy as np
from numexpr import evaluate

from . import v4_constants as v4c
from .utils import (
    MdfException,
    get_text_v4,
    SignalSource,
    UINT8_uf,
    UINT64_u,
    UINT64_uf,
    FLOAT64_u,
    sanitize_xml,
    block_fields,
)
from ..version import __version__


SEEK_START = v4c.SEEK_START
SEEK_END = v4c.SEEK_END
COMMON_SIZE = v4c.COMMON_SIZE
COMMON_u = v4c.COMMON_u
COMMON_uf = v4c.COMMON_uf

CN_BLOCK_SIZE = v4c.CN_BLOCK_SIZE
SIMPLE_CHANNEL_PARAMS_uf = v4c.SIMPLE_CHANNEL_PARAMS_uf


logger = logging.getLogger("asammdf")

__all__ = [
    "AttachmentBlock",
    "Channel",
    "ChannelArrayBlock",
    "ChannelGroup",
    "ChannelConversion",
    "DataBlock",
    "DataZippedBlock",
    "EventBlock",
    "FileIdentificationBlock",
    "HeaderBlock",
    "HeaderList",
    "DataList",
    "DataGroup",
    "FileHistory",
    "SourceInformation",
    "TextBlock",
]


class AttachmentBlock:
    """When adding new attachments only embedded attachments are allowed, with
    keyword argument *data* of type bytes

    *AttachmentBlock* has the following attributes, that are also available as
    dict like key-value pairs

    ATBLOCK fields

    * ``id`` - bytes : block ID; always b'##AT'
    * ``reserved0`` - int : reserved bytes
    * ``block_len`` - int : block bytes size
    * ``links_nr`` - int : number of links
    * ``next_at_addr`` - int : next ATBLOCK address
    * ``file_name_addr`` - int : address of TXBLOCK that contains the attachment
      file name
    * ``mime_addr`` - int : address of TXBLOCK that contains the attachment
      mime type description
    * ``comment_addr`` - int : address of TXBLOCK/MDBLOCK that contains the
      attachment comment
    * ``flags`` - int : ATBLOCK flags
    * ``creator_index`` - int : index of file history block
    * ``reserved1`` - int : reserved bytes
    * ``md5_sum`` - bytes : attachment file md5 sum
    * ``original_size`` - int : original uncompress file size in bytes
    * ``embedded_size`` - int : embedded compressed file size in bytes
    * ``embedded_data`` - bytes : embedded atatchment bytes

    Other attributes

    * ``address`` - int : attachment address
    * ``file_name`` - str : attachment file name
    * ``mime`` - str : mime type
    * ``comment`` - str : attachment comment

    Parameters
    ----------
    address : int
        block address; to be used for objects created from file
    stream : handle
        file handle; to be used for objects created from file
    for dynamically created objects :
        see the key-value pairs
    """

    __slots__ = (
        "address",
        "file_name",
        "mime",
        "comment",
        "id",
        "reserved0",
        "block_len",
        "links_nr",
        "next_at_addr",
        "file_name_addr",
        "mime_addr",
        "comment_addr",
        "flags",
        "creator_index",
        "reserved1",
        "md5_sum",
        "original_size",
        "embedded_size",
        "embedded_data",
    )

    def __init__(self, **kwargs):

        self.file_name = self.mime = self.comment = ""

        try:
            self.address = address = kwargs["address"]
            stream = kwargs["stream"]
            mapped = kwargs.get("mapped", False)

            if mapped:
                (
                    self.id,
                    self.reserved0,
                    self.block_len,
                    self.links_nr,
                    self.next_at_addr,
                    self.file_name_addr,
                    self.mime_addr,
                    self.comment_addr,
                    self.flags,
                    self.creator_index,
                    self.reserved1,
                    self.md5_sum,
                    self.original_size,
                    self.embedded_size,
                ) = v4c.AT_COMMON_uf(stream, address)

                address += v4c.AT_COMMON_SIZE

                self.embedded_data = stream[address: address + self.embedded_size]
            else:

                stream.seek(address)

                (
                    self.id,
                    self.reserved0,
                    self.block_len,
                    self.links_nr,
                    self.next_at_addr,
                    self.file_name_addr,
                    self.mime_addr,
                    self.comment_addr,
                    self.flags,
                    self.creator_index,
                    self.reserved1,
                    self.md5_sum,
                    self.original_size,
                    self.embedded_size,
                ) = v4c.AT_COMMON_u(stream.read(v4c.AT_COMMON_SIZE))

                self.embedded_data = stream.read(self.embedded_size)

            if self.id != b"##AT":
                message = f'Expected "##AT" block @{hex(address)} but found "{self.id}"'
                logger.exception(message)
                raise MdfException(message)

            self.file_name = get_text_v4(self.file_name_addr, stream, mapped=mapped)
            self.mime = get_text_v4(self.mime_addr, stream, mapped=mapped)
            self.comment = get_text_v4(self.comment_addr, stream, mapped=mapped)

        except KeyError:

            self.file_name = Path(kwargs.get("file_name", None) or "bin.bin")

            data = kwargs["data"]
            original_size = embedded_size = len(data)
            compression = kwargs.get("compression", False)
            embedded = kwargs.get("embedded", False)

            md5_sum = md5(data).digest()

            flags = v4c.FLAG_AT_MD5_VALID
            if embedded:
                flags |= v4c.FLAG_AT_EMBEDDED
                if compression:
                    flags |= v4c.FLAG_AT_COMPRESSED_EMBEDDED
                    data = compress(data)
                    embedded_size = len(data)

            else:
                self.file_name = Path(self.file_name.name)
                self.file_name.write_bytes(data)
                embedded_size = 0
                data = b""

            self.id = b"##AT"
            self.reserved0 = 0
            self.block_len = v4c.AT_COMMON_SIZE + embedded_size
            self.links_nr = 4
            self.next_at_addr = 0
            self.file_name_addr = 0
            self.mime_addr = 0
            self.comment_addr = 0
            self.flags = flags
            self.creator_index = 0
            self.reserved1 = 0
            self.md5_sum = md5_sum
            self.original_size = original_size
            self.embedded_size = embedded_size
            self.embedded_data = data

    def extract(self):
        """extract attachment data

        Returns
        -------
        data : bytes
        """
        if self.flags & v4c.FLAG_AT_EMBEDDED:
            if self.flags & v4c.FLAG_AT_COMPRESSED_EMBEDDED:
                data = decompress(self.embedded_data)
            else:
                data = self.embedded_data
            if self.flags & v4c.FLAG_AT_MD5_VALID:
                md5_worker = md5()
                md5_worker.update(data)
                md5_sum = md5_worker.digest()
                if self.md5_sum == md5_sum:
                    return data
                else:
                    message = f"ATBLOCK md5sum={self.md5_sum} and embedded data md5sum={md5_sum}"
                    logger.warning(message)
            else:
                return data
        else:
            logger.warning("external attachments not supported")

    def to_blocks(self, address, blocks, defined_texts):
        text = self.file_name
        if text:
            if text in defined_texts:
                self.file_name_addr = defined_texts[text]
            else:
                tx_block = TextBlock(text=str(text))
                self.file_name_addr = address
                defined_texts[text] = address
                tx_block.address = address
                address += tx_block.block_len
                blocks.append(tx_block)
        else:
            self.file_name_addr = 0

        text = self.mime
        if text:
            if text in defined_texts:
                self.mime_addr = defined_texts[text]
            else:
                tx_block = TextBlock(text=text)
                self.mime_addr = address
                defined_texts[text] = address
                tx_block.address = address
                address += tx_block.block_len
                blocks.append(tx_block)
        else:
            self.mime_addr = 0

        text = self.comment
        if text:
            if text in defined_texts:
                self.comment_addr = defined_texts[text]
            else:
                meta = text.startswith("<ATcomment")
                tx_block = TextBlock(text=text, meta=meta)
                self.comment_addr = address
                defined_texts[text] = address
                tx_block.address = address
                address += tx_block.block_len
                blocks.append(tx_block)
        else:
            self.comment_addr = 0

        blocks.append(self)
        self.address = address
        address += self.block_len

        align = address % 8
        if align % 8:
            blocks.append(b"\0" * (8 - align))
            address += 8 - align

        return address

    def __getitem__(self, item):
        return self.__getattribute__(item)

    def __setitem__(self, item, value):
        self.__setattr__(item, value)

    def __bytes__(self):
        fmt = f"{v4c.FMT_AT_COMMON}{self.embedded_size}s"
        result = pack(fmt, *[self[key] for key in v4c.KEYS_AT_BLOCK])
        return result


class Channel:
    """ If the `load_metadata` keyword argument is not provided or is False,
    then the conversion, source and display name information is not processed.
    Further more if the `parse_xml_comment` is not provided or is False, then
    the display name information from the channel comment is not processed (this
    is done to avoid expensive XML operations)

    *Channel* has the following attributes, that are also available as
    dict like key-value pairs

    CNBLOCK fields

    * ``id`` - bytes : block ID; always b'##CN'
    * ``reserved0`` - int : reserved bytes
    * ``block_len`` - int : block bytes size
    * ``links_nr`` - int : number of links
    * ``next_ch_addr`` - int : next ATBLOCK address
    * ``component_addr`` - int : address of first channel in case of structure channel
      composition, or ChannelArrayBlock in case of arrays
      file name
    * ``name_addr`` - int : address of TXBLOCK that contains the channel name
    * ``source_addr`` - int : address of channel source block
    * ``conversion_addr`` - int : address of channel conversion block
    * ``data_block_addr`` - int : address of signal data block for VLSD channels
    * ``unit_addr`` - int : address of TXBLOCK that contains the channel unit
    * ``comment_addr`` - int : address of TXBLOCK/MDBLOCK that contains the
      channel comment
    * ``attachment_<N>_addr`` - int : address of N-th ATBLOCK referenced by the
      current channel; if no ATBLOCK is referenced there will be no such key-value
      pair
    * ``default_X_dg_addr`` - int : address of DGBLOCK where the default X axis
      channel for the current channel is found; this key-value pair will not
      exist for channels that don't have a default X axis
    * ``default_X_cg_addr`` - int : address of CGBLOCK where the default X axis
      channel for the current channel is found; this key-value pair will not
      exist for channels that don't have a default X axis
    * ``default_X_ch_addr`` - int : address of default X axis
      channel for the current channel; this key-value pair will not
      exist for channels that don't have a default X axis
    * ``channel_type`` - int : integer code for the channel type
    * ``sync_type`` - int : integer code for the channel's sync type
    * ``data_type`` - int : integer code for the channel's data type
    * ``bit_offset`` - int : bit offset
    * ``byte_offset`` - int : byte offset within the data record
    * ``bit_count`` - int : channel bit count
    * ``flags`` - int : CNBLOCK flags
    * ``pos_invalidation_bit`` - int : invalidation bit position for the current
      channel if there are invalidation bytes in the data record
    * ``precision`` - int : integer code for teh precision
    * ``reserved1`` - int : reserved bytes
    * ``min_raw_value`` - int : min raw value of all samples
    * ``max_raw_value`` - int : max raw value of all samples
    * ``lower_limit`` - int : min physical value of all samples
    * ``upper_limit`` - int : max physical value of all samples
    * ``lower_ext_limit`` - int : min physical value of all samples
    * ``upper_ext_limit`` - int : max physical value of all samples

    Other attributes

    * ``address`` - int : channel address
    * ``attachments`` - list : list of referenced attachment blocks indexes;
      the index referece to the attachment block index
    * ``comment`` - str : channel comment
    * ``conversion`` - ChannelConversion : channel conversion; *None* if the
      channel has no conversion
    * ``display_name`` - str : channel display name; this is extracted from the
      XML channel comment
    * ``name`` - str : channel name
    * ``source`` - SourceInformation : channel source information; *None* if
      the channel has no source information
    * ``unit`` - str : channel unit

    Parameters
    ----------
    address : int
        block address; to be used for objects created from file
    stream : handle
        file handle; to be used for objects created from file
    load_metadata : bool
        option to load conversion, source and display_name; default *True*
    parse_xml_comment : bool
        option to parse XML channel comment to search for display name; default
        *True*
    for dynamically created objects :
        see the key-value pairs

    """

    __slots__ = (
        "name",
        "unit",
        "comment",
        "display_name",
        "conversion",
        "source",
        "attachment",
        "address",
        "dtype_fmt",
        "id",
        "reserved0",
        "block_len",
        "links_nr",
        "next_ch_addr",
        "component_addr",
        "name_addr",
        "source_addr",
        "conversion_addr",
        "data_block_addr",
        "unit_addr",
        "comment_addr",
        "channel_type",
        "sync_type",
        "data_type",
        "bit_offset",
        "byte_offset",
        "bit_count",
        "flags",
        "pos_invalidation_bit",
        "precision",
        "reserved1",
        "attachment_nr",
        "min_raw_value",
        "max_raw_value",
        "lower_limit",
        "upper_limit",
        "lower_ext_limit",
        "upper_ext_limit",
        "default_X_dg_addr",
        "default_X_cg_addr",
        "default_X_ch_addr",
        "attachment_addr",
    )

    def __init__(self, **kwargs):

        if "stream" in kwargs:

            self.address = address = kwargs["address"]
            stream = kwargs["stream"]
            mapped = kwargs.get('mapped', False)

            if mapped:

                (self.id, self.reserved0, self.block_len, self.links_nr) = COMMON_uf(stream, address)

                if self.id != b"##CN":
                    message = f'Expected "##CN" block @{hex(address)} but found "{self.id}"'
                    logger.exception(message)
                    raise MdfException(message)

                if self.block_len == CN_BLOCK_SIZE:

                    (
                        self.next_ch_addr,
                        self.component_addr,
                        self.name_addr,
                        self.source_addr,
                        self.conversion_addr,
                        self.data_block_addr,
                        self.unit_addr,
                        self.comment_addr,
                        self.channel_type,
                        self.sync_type,
                        self.data_type,
                        self.bit_offset,
                        self.byte_offset,
                        self.bit_count,
                        self.flags,
                        self.pos_invalidation_bit,
                        self.precision,
                        self.reserved1,
                        self.attachment_nr,
                        self.min_raw_value,
                        self.max_raw_value,
                        self.lower_limit,
                        self.upper_limit,
                        self.lower_ext_limit,
                        self.upper_ext_limit,
                    ) = SIMPLE_CHANNEL_PARAMS_uf(stream, address + COMMON_SIZE)

                else:

                    stream.seek(address + COMMON_SIZE)
                    block = stream.read(self.block_len - COMMON_SIZE)
                    links_nr = self.links_nr

                    links = unpack_from(f"<{links_nr}Q", block)
                    params = unpack_from(v4c.FMT_CHANNEL_PARAMS, block, links_nr * 8)

                    (
                        self.next_ch_addr,
                        self.component_addr,
                        self.name_addr,
                        self.source_addr,
                        self.conversion_addr,
                        self.data_block_addr,
                        self.unit_addr,
                        self.comment_addr,
                    ) = links[:8]

                    at_map = kwargs.get("at_map", {})
                    if params[10]:
                        self.attachment = []
                        self.attachment_addr = links[8]
                        self.attachment = at_map.get(links[8], 0)
                        self.links_nr -= params[10] - 1
                        self.block_len -= (params[10] - 1) * 8
                        params = list(params)
                        params[10] = 1

                    if params[6] & v4c.FLAG_CN_DEFAULT_X:
                        (
                            self.default_X_dg_addr,
                            self.default_X_cg_addr,
                            self.default_X_ch_addr,
                        ) = links[-3:]

                        # default X not supported yet
                        (
                            self.default_X_dg_addr,
                            self.default_X_cg_addr,
                            self.default_X_ch_addr,
                        ) = (0, 0, 0)

                    (
                        self.channel_type,
                        self.sync_type,
                        self.data_type,
                        self.bit_offset,
                        self.byte_offset,
                        self.bit_count,
                        self.flags,
                        self.pos_invalidation_bit,
                        self.precision,
                        self.reserved1,
                        self.attachment_nr,
                        self.min_raw_value,
                        self.max_raw_value,
                        self.lower_limit,
                        self.upper_limit,
                        self.lower_ext_limit,
                        self.upper_ext_limit,
                    ) = params

                self.name = get_text_v4(self.name_addr, stream, mapped=mapped)
                self.unit = get_text_v4(self.unit_addr, stream, mapped=mapped)
                self.comment = get_text_v4(self.comment_addr, stream, mapped=mapped)

                if kwargs.get("use_display_names", True):
                    try:
                        display_name = ET.fromstring(sanitize_xml(self.comment)).find(
                            ".//names/display"
                        )
                        if display_name is not None:
                            self.display_name = display_name.text
                    except:
                        self.display_name = ""
                else:
                    self.display_name = ""

                si_map = kwargs.get("si_map", {})
                cc_map = kwargs.get("cc_map", {})

                address = self.conversion_addr
                if address:
                    if mapped:
                        (size,) = UINT64_uf(stream, address + 8)
                        raw_bytes = stream[address: address + size]
                    else:
                        stream.seek(address + 8)
                        (size,) = UINT64_u(stream.read(8))
                        stream.seek(address)
                        raw_bytes = stream.read(size)
                    if raw_bytes in cc_map:
                        conv = cc_map[raw_bytes]
                    else:
                        conv = ChannelConversion(
                            raw_bytes=raw_bytes, stream=stream, address=address,
                            mapped=mapped,
                        )
                        cc_map[raw_bytes] = conv
                    self.conversion = conv
                else:
                    self.conversion = None

                address = self.source_addr
                if address:
                    if mapped:
                        raw_bytes = stream[address: address + v4c.SI_BLOCK_SIZE]
                    else:
                        stream.seek(address)
                        raw_bytes = stream.read(v4c.SI_BLOCK_SIZE)
                    if raw_bytes in si_map:
                        source = si_map[raw_bytes]
                    else:
                        source = SourceInformation(
                            raw_bytes=raw_bytes, stream=stream, address=address,
                            mapped=mapped,
                        )
                        si_map[raw_bytes] = source
                    self.source = source
                else:
                    self.source = None
                self.dtype_fmt = self.attachment = None
            else:
                stream.seek(address)

                block = stream.read(CN_BLOCK_SIZE)

                (self.id, self.reserved0, self.block_len, self.links_nr) = COMMON_uf(block)

                if self.id != b"##CN":
                    message = f'Expected "##CN" block @{hex(address)} but found "{self.id}"'
                    logger.exception(message)
                    raise MdfException(message)

                if self.block_len == CN_BLOCK_SIZE:

                    (
                        self.next_ch_addr,
                        self.component_addr,
                        self.name_addr,
                        self.source_addr,
                        self.conversion_addr,
                        self.data_block_addr,
                        self.unit_addr,
                        self.comment_addr,
                        self.channel_type,
                        self.sync_type,
                        self.data_type,
                        self.bit_offset,
                        self.byte_offset,
                        self.bit_count,
                        self.flags,
                        self.pos_invalidation_bit,
                        self.precision,
                        self.reserved1,
                        self.attachment_nr,
                        self.min_raw_value,
                        self.max_raw_value,
                        self.lower_limit,
                        self.upper_limit,
                        self.lower_ext_limit,
                        self.upper_ext_limit,
                    ) = SIMPLE_CHANNEL_PARAMS_uf(block, COMMON_SIZE)

                else:

                    block = block[24:] + stream.read(self.block_len - CN_BLOCK_SIZE)
                    links_nr = self.links_nr

                    links = unpack_from(f"<{links_nr}Q", block)
                    params = unpack_from(v4c.FMT_CHANNEL_PARAMS, block, links_nr * 8)

                    (
                        self.next_ch_addr,
                        self.component_addr,
                        self.name_addr,
                        self.source_addr,
                        self.conversion_addr,
                        self.data_block_addr,
                        self.unit_addr,
                        self.comment_addr,
                    ) = links[:8]

                    at_map = kwargs.get("at_map", {})
                    if params[10]:
                        self.attachment = []
                        self.attachment_addr = links[8]
                        self.attachment = at_map.get(links[8], 0)
                        self.links_nr -= params[10] - 1
                        self.block_len -= (params[10] - 1) * 8
                        params[10] = 1

                    if params[6] & v4c.FLAG_CN_DEFAULT_X:
                        (
                            self.default_X_dg_addr,
                            self.default_X_cg_addr,
                            self.default_X_ch_addr,
                        ) = links[-3:]

                        # default X not supported yet
                        (
                            self.default_X_dg_addr,
                            self.default_X_cg_addr,
                            self.default_X_ch_addr,
                        ) = (0, 0, 0)

                    (
                        self.channel_type,
                        self.sync_type,
                        self.data_type,
                        self.bit_offset,
                        self.byte_offset,
                        self.bit_count,
                        self.flags,
                        self.pos_invalidation_bit,
                        self.precision,
                        self.reserved1,
                        self.attachment_nr,
                        self.min_raw_value,
                        self.max_raw_value,
                        self.lower_limit,
                        self.upper_limit,
                        self.lower_ext_limit,
                        self.upper_ext_limit,
                    ) = params

                self.name = get_text_v4(self.name_addr, stream)
                self.unit = get_text_v4(self.unit_addr, stream)
                self.comment = get_text_v4(self.comment_addr, stream)

                if kwargs.get("use_display_names", True):
                    try:
                        display_name = ET.fromstring(sanitize_xml(self.comment)).find(
                            ".//names/display"
                        )
                        if display_name is not None:
                            self.display_name = display_name.text
                    except ET.ParseError:
                        self.display_name = ""
                else:
                    self.display_name = ""

                si_map = kwargs.get("si_map", {})
                cc_map = kwargs.get("cc_map", {})

                address = self.conversion_addr
                if address:
                    stream.seek(address + 8)
                    (size,) = UINT64_u(stream.read(8))
                    stream.seek(address)
                    raw_bytes = stream.read(size)
                    if raw_bytes in cc_map:
                        conv = cc_map[raw_bytes]
                    else:
                        conv = ChannelConversion(
                            raw_bytes=raw_bytes, stream=stream, address=address
                        )
                        cc_map[raw_bytes] = conv
                    self.conversion = conv
                else:
                    self.conversion = None

                address = self.source_addr
                if address:
                    stream.seek(address)
                    raw_bytes = stream.read(v4c.SI_BLOCK_SIZE)
                    if raw_bytes in si_map:
                        source = si_map[raw_bytes]
                    else:
                        source = SourceInformation(
                            raw_bytes=raw_bytes, stream=stream, address=address
                        )
                        si_map[raw_bytes] = source
                    self.source = source
                else:
                    self.source = None
                self.dtype_fmt = self.attachment = None
        else:
            self.address = 0
            self.name = self.comment = self.display_name = self.unit = ""
            self.conversion = self.source = self.attachment = self.dtype_fmt = None

            self.id = b"##CN"
            self.reserved0 = 0
            self.block_len = v4c.CN_BLOCK_SIZE
            self.links_nr = 8
            self.next_ch_addr = 0
            self.component_addr = 0
            self.name_addr = 0
            self.source_addr = 0
            self.conversion_addr = 0
            self.data_block_addr = 0
            self.unit_addr = 0
            self.comment_addr = 0
            try:
                self.attachment_addr = kwargs["attachment_addr"]
                self.block_len += 8
                self.links_nr += 1
                attachments = 1
            except KeyError:
                attachments = 0
            self.channel_type = kwargs["channel_type"]
            self.sync_type = kwargs.get("sync_type", 0)
            self.data_type = kwargs["data_type"]
            self.bit_offset = kwargs["bit_offset"]
            self.byte_offset = kwargs["byte_offset"]
            self.bit_count = kwargs["bit_count"]
            self.flags = kwargs.get("flags", 0)
            self.pos_invalidation_bit = kwargs.get("pos_invalidation_bit", 0)
            self.precision = kwargs.get("precision", 3)
            self.reserved1 = 0
            self.attachment_nr = attachments
            self.min_raw_value = kwargs.get("min_raw_value", 0)
            self.max_raw_value = kwargs.get("max_raw_value", 0)
            self.lower_limit = kwargs.get("lower_limit", 0)
            self.upper_limit = kwargs.get("upper_limit", 0)
            self.lower_ext_limit = kwargs.get("lower_ext_limit", 0)
            self.upper_ext_limit = kwargs.get("upper_ext_limit", 0)

        # ignore MLSD signal data
        if self.channel_type == v4c.CHANNEL_TYPE_MLSD:
            self.data_block_addr = 0
            self.channel_type = v4c.CHANNEL_TYPE_VALUE

    def __getitem__(self, item):
        return self.__getattribute__(item)

    def __setitem__(self, item, value):
        self.__setattr__(item, value)

    def to_blocks(self, address, blocks, defined_texts, cc_map, si_map):
        text = self.name
        if text:
            if text in defined_texts:
                self.name_addr = defined_texts[text]
            else:
                tx_block = TextBlock(text=text)
                self.name_addr = address
                defined_texts[text] = address
                tx_block.address = address
                address += tx_block.block_len
                blocks.append(tx_block)
        else:
            self.name_addr = 0

        text = self.unit
        if text:
            if text in defined_texts:
                self.unit_addr = defined_texts[text]
            else:
                tx_block = TextBlock(text=text)
                self.unit_addr = address
                defined_texts[text] = address
                tx_block.address = address
                address += tx_block.block_len
                blocks.append(tx_block)
        else:
            self.unit_addr = 0

        comment = self.comment
        display_name = self.display_name

        if display_name and not text:
            text = v4c.CN_COMMENT_TEMPLATE.format(comment, display_name)
        elif display_name and comment:
            if not comment.startswith("<CNcomment"):
                text = v4c.CN_COMMENT_TEMPLATE.format(comment, display_name)
            else:
                if display_name not in comment:
                    try:
                        CNcomment = ET.fromstring(comment)
                        display_name_element = CNcomment.find(".//names/display")
                        if display_name is not None:
                            display_name_element.text = display_name
                        else:

                            display = ET.Element("display")
                            display.text = display_name
                            names = ET.Element("names")
                            names.append(display)
                            CNcomment.append(names)

                        text = ET.tostring(CNcomment).decode("utf-8")

                    except UnicodeEncodeError:
                        text = comment
                else:
                    text = comment
        else:
            text = comment

        if text:
            if text in defined_texts:
                self.comment_addr = defined_texts[text]
            else:
                meta = text.startswith("<CNcomment")
                tx_block = TextBlock(text=text, meta=meta)
                self.comment_addr = address
                defined_texts[text] = address
                tx_block.address = address
                address += tx_block.block_len
                blocks.append(tx_block)
        else:
            self.comment_addr = 0

        conversion = self.conversion
        if conversion:
            address = conversion.to_blocks(address, blocks, defined_texts, cc_map)
            self.conversion_addr = conversion.address
        else:
            self.conversion_addr = 0

        source = self.source
        if source:
            address = source.to_blocks(address, blocks, defined_texts, si_map)
            self.source_addr = source.address
        else:
            self.source_addr = 0

        blocks.append(self)
        self.address = address
        address += self.block_len

        return address

    def __bytes__(self):

        if self.block_len == v4c.CN_BLOCK_SIZE:
            return v4c.SIMPLE_CHANNEL_PACK(
                self.id,
                self.reserved0,
                self.block_len,
                self.links_nr,
                self.next_ch_addr,
                self.component_addr,
                self.name_addr,
                self.source_addr,
                self.conversion_addr,
                self.data_block_addr,
                self.unit_addr,
                self.comment_addr,
                self.channel_type,
                self.sync_type,
                self.data_type,
                self.bit_offset,
                self.byte_offset,
                self.bit_count,
                self.flags,
                self.pos_invalidation_bit,
                self.precision,
                self.reserved1,
                self.attachment_nr,
                self.min_raw_value,
                self.max_raw_value,
                self.lower_limit,
                self.upper_limit,
                self.lower_ext_limit,
                self.upper_ext_limit,
            )
        else:
            fmt = v4c.FMT_CHANNEL.format(self.links_nr)

            keys = (
                "id",
                "reserved0",
                "block_len",
                "links_nr",
                "next_ch_addr",
                "component_addr",
                "name_addr",
                "source_addr",
                "conversion_addr",
                "data_block_addr",
                "unit_addr",
                "comment_addr",
            )
            if self.attachment_nr:
                keys += ("attachment_addr",)

            if self.flags & v4c.FLAG_CN_DEFAULT_X:
                keys += ("default_X_dg_addr", "default_X_cg_addr", "default_X_ch_addr")
            keys += (
                "channel_type",
                "sync_type",
                "data_type",
                "bit_offset",
                "byte_offset",
                "bit_count",
                "flags",
                "pos_invalidation_bit",
                "precision",
                "reserved1",
                "attachment_nr",
                "min_raw_value",
                "max_raw_value",
                "lower_limit",
                "upper_limit",
                "lower_ext_limit",
                "upper_ext_limit",
            )
            return pack(fmt, *[getattr(self, key) for key in keys])

    def __str__(self):
        return f"""<Channel (name: {self.name}, unit: {self.unit}, comment: {self.comment}, address: {hex(self.address)},
    conversion: {self.conversion},
    source: {self.source},
    fields: {', '.join(block_fields(self))})>"""

    def metadata(self):
        if self.block_len == v4c.CN_BLOCK_SIZE:
            keys = v4c.KEYS_SIMPLE_CHANNEL
        else:
            keys = (
                "id",
                "reserved0",
                "block_len",
                "links_nr",
                "next_ch_addr",
                "component_addr",
                "name_addr",
                "source_addr",
                "conversion_addr",
                "data_block_addr",
                "unit_addr",
                "comment_addr",
            )
            if self.attachment_nr:
                keys += ("attachment_addr",)

            if self.flags & v4c.FLAG_CN_DEFAULT_X:
                keys += ("default_X_dg_addr", "default_X_cg_addr", "default_X_ch_addr")
            keys += (
                "channel_type",
                "sync_type",
                "data_type",
                "bit_offset",
                "byte_offset",
                "bit_count",
                "flags",
                "pos_invalidation_bit",
                "precision",
                "reserved1",
                "attachment_nr",
                "min_raw_value",
                "max_raw_value",
                "lower_limit",
                "upper_limit",
                "lower_ext_limit",
                "upper_ext_limit",
            )

        max_len = max(len(key) for key in keys)
        template = f"{{: <{max_len}}}: {{}}"

        metadata = []
        lines = f"""
name: {self.name}
display name: {self.display_name}
address: {hex(self.address)}
comment: {self.comment}

""".split("\n")

        for key in keys:
            val = getattr(self, key)
            if key.endswith("addr") or key.startswith("text_"):
                lines.append(template.format(key, hex(val)))
            elif isinstance(val, float):
                lines.append(template.format(key, round(val, 6)))
            else:
                if isinstance(val, bytes):
                    lines.append(template.format(key, val.strip(b"\0")))
                else:
                    lines.append(template.format(key, val))
        for line in lines:
            if not line:
                metadata.append(line)
            else:
                for wrapped_line in wrap(line, width=120):
                    metadata.append(wrapped_line)

        return "\n".join(metadata)

    def __contains__(self, item):
        return hasattr(self, item)

    def __lt__(self, other):
        self_byte_offset = self.byte_offset
        other_byte_offset = other.byte_offset

        if self_byte_offset < other_byte_offset:
            result = 1
        elif self_byte_offset == other_byte_offset:
            self_range = self.bit_offset + self.bit_count
            other_range = other.bit_offset + other.bit_count

            if self_range > other_range:
                result = 1
            else:
                result = 0
        else:
            result = 0
        return result


class _ChannelArrayBlockBase:
    __slots__ = (
        "referenced_channels",
        "address",
        "id",
        "reserved0",
        "block_len",
        "links_nr",
        "ca_type",
        "storage",
        "dims",
        "flags",
        "byte_offset_base",
        "invalidation_bit_base",
    )


class ChannelArrayBlock(_ChannelArrayBlockBase):
    """
    Other attributes

    * ``address`` - int : array block address
    * ``referenced_channels`` - list : list of (group index, channel index)
      pairs referenced by this array block

    """

    def __init__(self, **kwargs):

        self.referenced_channels = []

        try:
            self.address = address = kwargs["address"]
            stream = kwargs["stream"]

            mapped = kwargs.get("mapped", False)

            if mapped:

                (self.id, self.reserved0, self.block_len, self.links_nr) = v4c.COMMON_uf(
                    stream, address
                )

                if self.id != b"##CA":
                    message = f'Expected "##CA" block @{hex(address)} but found "{self.id}"'
                    logger.exception(message)
                    raise MdfException(message)

                nr = self.links_nr
                address += COMMON_SIZE
                links = unpack_from(f"<{nr}Q", stream, address)
                self.composition_addr = links[0]

                address += nr * 8
                values = unpack_from("<2BHIiI", stream, address)
                dims_nr = values[2]

                if nr == 1:
                    pass

                # lookup table with fixed axis
                elif nr == dims_nr + 1:
                    for i in range(dims_nr):
                        self[f"axis_conversion_{i}"] = links[i + 1]

                # lookup table with CN template
                elif nr == 4 * dims_nr + 1:
                    for i in range(dims_nr):
                        self[f"axis_conversion_{i}"] = links[i + 1]
                    links = links[dims_nr + 1 :]
                    for i in range(dims_nr):
                        self[f"scale_axis_{i}_dg_addr"] = links[3 * i]
                        self[f"scale_axis_{i}_cg_addr"] = links[3 * i + 1]
                        self[f"scale_axis_{i}_ch_addr"] = links[3 * i + 2]

                (
                    self.ca_type,
                    self.storage,
                    self.dims,
                    self.flags,
                    self.byte_offset_base,
                    self.invalidation_bit_base,
                ) = values

                address += 16
                dim_sizes = unpack_from(f"<{dims_nr}Q", stream, address)
                for i, size in enumerate(dim_sizes):
                    self[f"dim_size_{i}"] = size

                if self.flags & v4c.FLAG_CA_FIXED_AXIS:
                    for i in range(dims_nr):
                        for j in range(self[f"dim_size_{i}"]):
                            (value,) = FLOAT64_u(stream.read(8))
                            self[f"axis_{i}_value_{j}"] = value
            else:

                stream.seek(address)

                (self.id, self.reserved0, self.block_len, self.links_nr) = unpack(
                    "<4sI2Q", stream.read(24)
                )

                if self.id != b"##CA":
                    message = f'Expected "##CA" block @{hex(address)} but found "{self.id}"'
                    logger.exception(message)
                    raise MdfException(message)

                nr = self.links_nr
                links = unpack(f"<{nr}Q", stream.read(8 * nr))
                self.composition_addr = links[0]

                values = unpack("<2BHIiI", stream.read(16))
                dims_nr = values[2]

                if nr == 1:
                    pass

                # lookup table with fixed axis
                elif nr == dims_nr + 1:
                    for i in range(dims_nr):
                        self[f"axis_conversion_{i}"] = links[i + 1]

                # lookup table with CN template
                elif nr == 4 * dims_nr + 1:
                    for i in range(dims_nr):
                        self[f"axis_conversion_{i}"] = links[i + 1]
                    links = links[dims_nr + 1 :]
                    for i in range(dims_nr):
                        self[f"scale_axis_{i}_dg_addr"] = links[3 * i]
                        self[f"scale_axis_{i}_cg_addr"] = links[3 * i + 1]
                        self[f"scale_axis_{i}_ch_addr"] = links[3 * i + 2]

                (
                    self.ca_type,
                    self.storage,
                    self.dims,
                    self.flags,
                    self.byte_offset_base,
                    self.invalidation_bit_base,
                ) = values

                dim_sizes = unpack(f"<{dims_nr}Q", stream.read(8 * dims_nr))
                for i, size in enumerate(dim_sizes):
                    self[f"dim_size_{i}"] = size

                if self.flags & v4c.FLAG_CA_FIXED_AXIS:
                    for i in range(dims_nr):
                        for j in range(self[f"dim_size_{i}"]):
                            (value,) = FLOAT64_u(stream.read(8))
                            self[f"axis_{i}_value_{j}"] = value

        except KeyError:
            self.id = b"##CA"
            self.reserved0 = 0

            ca_type = kwargs["ca_type"]

            if ca_type == v4c.CA_TYPE_ARRAY:
                dims_nr = kwargs["dims"]
                self.block_len = 48 + dims_nr * 8
                self.links_nr = 1
                self.composition_addr = 0
                self.ca_type = v4c.CA_TYPE_ARRAY
                self.storage = v4c.CA_STORAGE_TYPE_CN_TEMPLATE
                self.dims = dims_nr
                self.flags = 0
                self.byte_offset_base = kwargs.get("byte_offset_base", 1)
                self.invalidation_bit_base = kwargs.get("invalidation_bit_base", 0)
                for i in range(dims_nr):
                    self[f"dim_size_{i}"] = kwargs[f"dim_size_{i}"]
            elif ca_type == v4c.CA_TYPE_SCALE_AXIS:
                self.block_len = 56
                self.links_nr = 1
                self.composition_addr = 0
                self.ca_type = v4c.CA_TYPE_SCALE_AXIS
                self.storage = v4c.CA_STORAGE_TYPE_CN_TEMPLATE
                self.dims = 1
                self.flags = 0
                self.byte_offset_base = kwargs.get("byte_offset_base", 1)
                self.invalidation_bit_base = kwargs.get("invalidation_bit_base", 0)
                self.dim_size_0 = kwargs["dim_size_0"]
            elif ca_type == v4c.CA_TYPE_LOOKUP:
                flags = kwargs["flags"]
                dims_nr = kwargs["dims"]
                values = sum(kwargs[f"dim_size_{i}"] for i in range(dims_nr))
                if flags & v4c.FLAG_CA_FIXED_AXIS:
                    self.block_len = 48 + dims_nr * 16 + values * 8
                    self.links_nr = 1 + dims_nr
                    self.composition_addr = 0
                    for i in range(dims_nr):
                        self[f"axis_conversion_{i}"] = 0
                    self.ca_type = v4c.CA_TYPE_LOOKUP
                    self.storage = v4c.CA_STORAGE_TYPE_CN_TEMPLATE
                    self.dims = dims_nr
                    self.flags = v4c.FLAG_CA_FIXED_AXIS | v4c.FLAG_CA_AXIS
                    self.byte_offset_base = kwargs.get("byte_offset_base", 1)
                    self.invalidation_bit_base = kwargs.get("invalidation_bit_base", 0)
                    for i in range(dims_nr):
                        self[f"dim_size_{i}"] = kwargs[f"dim_size_{i}"]
                    for i in range(dims_nr):
                        for j in range(self[f"dim_size_{i}"]):
                            self[f"axis_{i}_value_{j}"] = kwargs.get(
                                f"axis_{i}_value_{j}", j
                            )
                else:
                    self.block_len = 48 + dims_nr * 5 * 8
                    self.links_nr = 1 + dims_nr * 4
                    self.composition_addr = 0
                    for i in range(dims_nr):
                        self[f"axis_conversion_{i}"] = 0
                    for i in range(dims_nr):
                        self[f"scale_axis_{i}_dg_addr"] = 0
                        self[f"scale_axis_{i}_cg_addr"] = 0
                        self[f"scale_axis_{i}_ch_addr"] = 0
                    self.ca_type = v4c.CA_TYPE_LOOKUP
                    self.storage = v4c.CA_STORAGE_TYPE_CN_TEMPLATE
                    self.dims = dims_nr
                    self.flags = v4c.FLAG_CA_AXIS
                    self.byte_offset_base = kwargs.get("byte_offset_base", 1)
                    self.invalidation_bit_base = kwargs.get("invalidation_bit_base", 0)
                    for i in range(dims_nr):
                        self[f"dim_size_{i}"] = kwargs[f"dim_size_{i}"]

    def __getitem__(self, item):
        return self.__getattribute__(item)

    def __setitem__(self, item, value):
        self.__setattr__(item, value)

    def __str__(self):
        return "<ChannelArrayBlock (referenced channels: {}, address: {}, fields: {})>".format(
            self.referenced_channels, hex(self.address), dict(self)
        )

    def __bytes__(self):
        flags = self.flags
        ca_type = self.ca_type
        dims_nr = self.dims

        if ca_type == v4c.CA_TYPE_ARRAY:
            keys = (
                "id",
                "reserved0",
                "block_len",
                "links_nr",
                "composition_addr",
                "ca_type",
                "storage",
                "dims",
                "flags",
                "byte_offset_base",
                "invalidation_bit_base",
            )
            keys += tuple(f"dim_size_{i}" for i in range(dims_nr))
            fmt = f"<4sI3Q2BHIiI{dims_nr}Q"
        elif ca_type == v4c.CA_TYPE_SCALE_AXIS:
            keys = (
                "id",
                "reserved0",
                "block_len",
                "links_nr",
                "composition_addr",
                "ca_type",
                "storage",
                "dims",
                "flags",
                "byte_offset_base",
                "invalidation_bit_base",
                "dim_size_0",
            )

            fmt = "<4sI3Q2BHIiIQ"
        elif ca_type == v4c.CA_TYPE_LOOKUP:
            if flags & v4c.FLAG_CA_FIXED_AXIS:
                nr = sum(self[f"dim_size_{i}"] for i in range(dims_nr))
                keys = ("id", "reserved0", "block_len", "links_nr", "composition_addr")
                keys += tuple(f"axis_conversion_{i}" for i in range(dims_nr))
                keys += (
                    "ca_type",
                    "storage",
                    "dims",
                    "flags",
                    "byte_offset_base",
                    "invalidation_bit_base",
                )
                keys += tuple(f"dim_size_{i}" for i in range(dims_nr))
                keys += tuple(
                    f"axis_{i}_value_{j}"
                    for i in range(dims_nr)
                    for j in range(self[f"dim_size_{i}"])
                )
                fmt = "<4sI{}Q2BHIiI{}Q{}d"
                fmt = fmt.format(self.links_nr + 2, dims_nr, nr)
            else:
                keys = ("id", "reserved0", "block_len", "links_nr", "composition_addr")
                keys += tuple(f"axis_conversion_{i}" for i in range(dims_nr))
                for i in range(dims_nr):
                    keys += (
                        f"scale_axis_{i}_dg_addr",
                        f"scale_axis_{i}_cg_addr",
                        f"scale_axis_{i}_ch_addr",
                    )
                keys += (
                    "ca_type",
                    "storage",
                    "dims",
                    "flags",
                    "byte_offset_base",
                    "invalidation_bit_base",
                )
                keys += tuple(f"dim_size_{i}" for i in range(dims_nr))
                fmt = "<4sI{}Q2BHIiI{}Q".format(self.links_nr + 2, dims_nr)

        result = pack(fmt, *[getattr(self, key) for key in keys])
        return result


class ChannelGroup:
    """*ChannelGroup* has the following attributes, that are also available as
    dict like key-value pairs

    CGBLOCK fields

    * ``id`` - bytes : block ID; always b'##CG'
    * ``reserved0`` - int : reserved bytes
    * ``block_len`` - int : block bytes size
    * ``links_nr`` - int : number of links
    * ``next_cg_addr`` - int : next channel group address
    * ``first_ch_addr`` - int : address of first channel of this channel group
    * ``acq_name_addr`` - int : address of TextBLock that contains the channel
      group acquisition name
    * ``acq_source_addr`` - int : addres of SourceInformation that contains the
      channel group source
    * ``first_sample_reduction_addr`` - int : address of first SRBLOCK; this is
      considered 0 since sample reduction is not yet supported
    * ``comment_addr`` - int : address of TXBLOCK/MDBLOCK that contains the
      channel group comment
    * ``record_id`` - int : record ID for thei channel group
    * ``cycles_nr`` - int : number of cycles for this channel group
    * ``flags`` - int : channel group flags
    * ``path_separator`` - int : ordinal for character used as path separator
    * ``reserved1`` - int : reserved bytes
    * ``samples_byte_nr`` - int : number of bytes used for channels samples in
      the record for this channel group; this does not contain the invalidation
      bytes
    * ``invalidation_bytes_nr`` - int : number of bytes used for invalidation
      bits by this channl group

    Other attributes

    * ``acq_name`` - str : acquisition name
    * ``acq_source`` - SourceInformation : acquisition source information
    * ``address`` - int : channel group address
    * ``comment`` - str : channel group comment

    """

    __slots__ = (
        "address",
        "acq_name",
        "acq_source",
        "comment",
        "name",
        "id",
        "reserved0",
        "block_len",
        "links_nr",
        "next_cg_addr",
        "first_ch_addr",
        "acq_name_addr",
        "acq_source_addr",
        "first_sample_reduction_addr",
        "comment_addr",
        "record_id",
        "cycles_nr",
        "flags",
        "path_separator",
        "reserved1",
        "samples_byte_nr",
        "invalidation_bytes_nr",
    )

    def __init__(self, **kwargs):

        self.acq_name = self.comment = ""
        self.acq_source = None

        try:
            self.address = address = kwargs["address"]
            stream = kwargs["stream"]
            mapped = kwargs.get("mapped", False)

            if mapped:
                (
                    self.id,
                    self.reserved0,
                    self.block_len,
                    self.links_nr,
                    self.next_cg_addr,
                    self.first_ch_addr,
                    self.acq_name_addr,
                    self.acq_source_addr,
                    self.first_sample_reduction_addr,
                    self.comment_addr,
                    self.record_id,
                    self.cycles_nr,
                    self.flags,
                    self.path_separator,
                    self.reserved1,
                    self.samples_byte_nr,
                    self.invalidation_bytes_nr,
                ) = v4c.CHANNEL_GROUP_uf(stream, address)
            else:

                stream.seek(address)

                (
                    self.id,
                    self.reserved0,
                    self.block_len,
                    self.links_nr,
                    self.next_cg_addr,
                    self.first_ch_addr,
                    self.acq_name_addr,
                    self.acq_source_addr,
                    self.first_sample_reduction_addr,
                    self.comment_addr,
                    self.record_id,
                    self.cycles_nr,
                    self.flags,
                    self.path_separator,
                    self.reserved1,
                    self.samples_byte_nr,
                    self.invalidation_bytes_nr,
                ) = v4c.CHANNEL_GROUP_u(stream.read(v4c.CG_BLOCK_SIZE))

            if self.id != b"##CG":
                message = f'Expected "##CG" block @{hex(address)} but found "{self.id}"'

                logger.exception(message)
                raise MdfException(message)

            self.acq_name = get_text_v4(self.acq_name_addr, stream, mapped=mapped)
            self.comment = get_text_v4(self.comment_addr, stream, mapped=mapped)

            if self.acq_source_addr:
                self.acq_source = SourceInformation(
                    address=self.acq_source_addr, stream=stream, mapped=mapped
                )

        except KeyError:
            self.address = 0
            self.id = b"##CG"
            self.reserved0 = kwargs.get("reserved0", 0)
            self.block_len = kwargs.get("block_len", v4c.CG_BLOCK_SIZE)
            self.links_nr = kwargs.get("links_nr", 6)
            self.next_cg_addr = kwargs.get("next_cg_addr", 0)
            self.first_ch_addr = kwargs.get("first_ch_addr", 0)
            self.acq_name_addr = kwargs.get("acq_name_addr", 0)
            self.acq_source_addr = kwargs.get("acq_source_addr", 0)
            self.first_sample_reduction_addr = kwargs.get(
                "first_sample_reduction_addr", 0
            )
            self.comment_addr = kwargs.get("comment_addr", 0)
            self.record_id = kwargs.get("record_id", 1)
            self.cycles_nr = kwargs.get("cycles_nr", 0)
            self.flags = kwargs.get("flags", 0)
            self.path_separator = kwargs.get("path_separator", 0)
            self.reserved1 = kwargs.get("reserved1", 0)
            self.samples_byte_nr = kwargs.get("samples_byte_nr", 0)
            self.invalidation_bytes_nr = kwargs.get("invalidation_bytes_nr", 0)

    def __getitem__(self, item):
        return self.__getattribute__(item)

    def __setitem__(self, item, value):
        self.__setattr__(item, value)

    def to_blocks(self, address, blocks, defined_texts, si_map):
        text = self.acq_name
        if text:
            if text in defined_texts:
                self.acq_name_addr = defined_texts[text]
            else:
                tx_block = TextBlock(text=text)
                self.acq_name_addr = address
                defined_texts[text] = address
                tx_block.address = address
                address += tx_block.block_len
                blocks.append(tx_block)
        else:
            self.acq_name_addr = 0

        text = self.comment
        if text:
            if text in defined_texts:
                self.comment_addr = defined_texts[text]
            else:
                meta = text.startswith("<CGcomment")
                tx_block = TextBlock(text=text, meta=meta)
                self.comment_addr = address
                defined_texts[text] = address
                tx_block.address = address
                address += tx_block.block_len
                blocks.append(tx_block)
        else:
            self.comment_addr = 0

        source = self.acq_source
        if source:
            address = source.to_blocks(address, blocks, defined_texts, si_map)
            self.acq_source_addr = source.address
        else:
            self.acq_source_addr = 0

        blocks.append(self)
        self.address = address
        address += self.block_len

        return address

    def __bytes__(self):
        result = v4c.CHANNEL_GROUP_p(
            self.id,
            self.reserved0,
            self.block_len,
            self.links_nr,
            self.next_cg_addr,
            self.first_ch_addr,
            self.acq_name_addr,
            self.acq_source_addr,
            self.first_sample_reduction_addr,
            self.comment_addr,
            self.record_id,
            self.cycles_nr,
            self.flags,
            self.path_separator,
            self.reserved1,
            self.samples_byte_nr,
            self.invalidation_bytes_nr,
        )
        return result


class _ChannelConversionBase:
    __slots__ = (
        "name",
        "unit",
        "comment",
        "formula",
        "referenced_blocks",
        "address",
        "id",
        "reserved0",
        "block_len",
        "links_nr",
        "name_addr",
        "unit_addr",
        "comment_addr",
        "inv_conv_addr",
        "conversion_type",
        "precision",
        "flags",
        "ref_param_nr",
        "val_param_nr",
        "min_phy_value",
        "max_phy_value",
        "a",
        "b",
        "P1",
        "P2",
        "P3",
        "P4",
        "P5",
        "P6",
    )


class ChannelConversion(_ChannelConversionBase):
    """*ChannelConversion* has the following attributes, that are also available as
    dict like key-value pairs

    CCBLOCK common fields

    * ``id`` - bytes : block ID; always b'##CG'
    * ``reserved0`` - int : reserved bytes
    * ``block_len`` - int : block bytes size
    * ``links_nr`` - int : number of links
    * ``name_addr`` - int : address of TXBLOCK that contains the
      conversion name
    * ``unit_addr`` - int : address of TXBLOCK that contains the
      conversion unit
    * ``comment_addr`` - int : address of TXBLOCK/MDBLOCK that contains the
      conversion comment
    * ``inv_conv_addr`` int : address of invers conversion
    * ``conversion_type`` int : integer code for conversion type
    * ``precision`` - int : integer code for precision
    * ``flags`` - int : conversion block flags
    * ``ref_param_nr`` - int : number fo referenced parameters (linked
      parameters)
    * ``val_param_nr`` - int : number of value parameters
    * ``min_phy_value`` - float : minimum physical channel value
    * ``max_phy_value`` - float : maximum physical channel value

    CCBLOCK specific fields

    * linear conversion

        * ``a`` - float : factor
        * ``b`` - float : offset

    * rational conversion

        * ``P1`` to ``P6`` - float : parameters

    * algebraic conversion

        * ``formula_addr`` - address of TXBLOCK that contains the
          the algebraic conversion formula

    * tabluar conversion with or without interpolation

        * ``raw_<N>`` - float : N-th raw value
        * ``phys_<N>`` - float : N-th physical value

    * tabular range conversion

        * ``lower_<N>`` - float : N-th lower value
        * ``upper_<N>`` - float : N-th upper value
        * ``phys_<N>`` - float : N-th physical value

    * tabular value to text conversion

        * ``val_<N>`` - float : N-th raw value
        * ``text_<N>`` - int : address of N-th TXBLOCK that
          contains the physical value
        * ``default`` - int : address of TXBLOCK that contains
          the default physical value

    * tabular range to text conversion

        * ``lower_<N>`` - float : N-th lower value
        * ``upper_<N>`` - float : N-th upper value
        * ``text_<N>`` - int : address of N-th TXBLOCK that
          contains the physical value
        * ``default`` - int : address of TXBLOCK that contains
          the default physical value

    * text to value conversion

        * ``val_<N>`` - float : N-th physical value
        * ``text_<N>`` - int : address of N-th TXBLOCK that
          contains the raw value
        * ``val_default`` - float : default physical value

    * text tranfosrmation (translation) conversion

        * ``input_<N>_addr`` - int : address of N-th TXBLOCK that
          contains the raw value
        * ``output_<N>_addr`` - int : address of N-th TXBLOCK that
          contains the physical value
        * ``default_addr`` - int : address of TXBLOCK that contains
          the default physical value

    Other attributes

    * ``address`` - int : channel conversion address
    * ``comment`` - str : channel conversion comment
    * ``formula`` - str : algebraic conversion formula; default ''
    * ``referenced_blocks`` - list :  list of refenced blocks; can be TextBlock
      objects for value to text, and text to text conversions; for partial
      conversions the referenced blocks can be ChannelConversion obejct as well
    * ``name`` - str : channel conversion name
    * ``unit`` - str : channel conversion unit

    """

    def __init__(self, **kwargs):
        self.name = self.unit = self.comment = self.formula = ""
        self.referenced_blocks = None

        if "stream" in kwargs:
            mapped = kwargs.get("mapped", False)

            stream = kwargs["stream"]
            try:
                self.address = address = kwargs.get("address", 0)
                block = kwargs["raw_bytes"]
                (self.id, self.reserved0, self.block_len, self.links_nr) = COMMON_uf(
                    block
                )

                if self.id != b"##CC":
                    message = f'Expected "##CC" block @{hex(address)} but found "{self.id}"'
                    logger.exception(message)
                    raise MdfException(message)

                block = block[COMMON_SIZE:]

            except KeyError:
                self.address = address = kwargs["address"]
                stream.seek(address)

                (self.id, self.reserved0, self.block_len, self.links_nr) = COMMON_u(
                    stream.read(COMMON_SIZE)
                )

                if self.id != b"##CC":
                    message = f'Expected "##CC" block @{hex(address)} but found "{self.id}"'
                    logger.exception(message)
                    raise MdfException(message)

                block = stream.read(self.block_len - COMMON_SIZE)

            (conv,) = UINT8_uf(block, self.links_nr * 8)

            if conv == v4c.CONVERSION_TYPE_NON:
                (
                    self.name_addr,
                    self.unit_addr,
                    self.comment_addr,
                    self.inv_conv_addr,
                    self.conversion_type,
                    self.precision,
                    self.flags,
                    self.ref_param_nr,
                    self.val_param_nr,
                    self.min_phy_value,
                    self.max_phy_value,
                ) = v4c.CONVERSION_NONE_INIT_u(block)

            elif conv == v4c.CONVERSION_TYPE_LIN:
                (
                    self.name_addr,
                    self.unit_addr,
                    self.comment_addr,
                    self.inv_conv_addr,
                    self.conversion_type,
                    self.precision,
                    self.flags,
                    self.ref_param_nr,
                    self.val_param_nr,
                    self.min_phy_value,
                    self.max_phy_value,
                    self.b,
                    self.a,
                ) = v4c.CONVERSION_LINEAR_INIT_u(block)

            elif conv == v4c.CONVERSION_TYPE_RAT:
                (
                    self.name_addr,
                    self.unit_addr,
                    self.comment_addr,
                    self.inv_conv_addr,
                    self.conversion_type,
                    self.precision,
                    self.flags,
                    self.ref_param_nr,
                    self.val_param_nr,
                    self.min_phy_value,
                    self.max_phy_value,
                    self.P1,
                    self.P2,
                    self.P3,
                    self.P4,
                    self.P5,
                    self.P6,
                ) = unpack(v4c.FMT_CONVERSION_RAT_INIT, block)

            elif conv == v4c.CONVERSION_TYPE_ALG:
                (
                    self.name_addr,
                    self.unit_addr,
                    self.comment_addr,
                    self.inv_conv_addr,
                    self.formula_addr,
                    self.conversion_type,
                    self.precision,
                    self.flags,
                    self.ref_param_nr,
                    self.val_param_nr,
                    self.min_phy_value,
                    self.max_phy_value,
                ) = unpack(v4c.FMT_CONVERSION_ALGEBRAIC_INIT, block)

            elif conv in (v4c.CONVERSION_TYPE_TABI, v4c.CONVERSION_TYPE_TAB):
                (
                    self.name_addr,
                    self.unit_addr,
                    self.comment_addr,
                    self.inv_conv_addr,
                    self.conversion_type,
                    self.precision,
                    self.flags,
                    self.ref_param_nr,
                    self.val_param_nr,
                    self.min_phy_value,
                    self.max_phy_value,
                ) = unpack_from(v4c.FMT_CONVERSION_NONE_INIT, block)

                nr = self.val_param_nr
                values = unpack(f"<{nr}d", block[56:])
                for i in range(nr // 2):
                    self[f"raw_{i}"], self[f"phys_{i}"] = (
                        values[i * 2],
                        values[2 * i + 1],
                    )

            elif conv == v4c.CONVERSION_TYPE_RTAB:
                (
                    self.name_addr,
                    self.unit_addr,
                    self.comment_addr,
                    self.inv_conv_addr,
                    self.conversion_type,
                    self.precision,
                    self.flags,
                    self.ref_param_nr,
                    self.val_param_nr,
                    self.min_phy_value,
                    self.max_phy_value,
                ) = unpack_from(v4c.FMT_CONVERSION_NONE_INIT, block)
                nr = self.val_param_nr
                values = unpack(f"<{nr}d", block[56:])
                for i in range((nr - 1) // 3):
                    (self[f"lower_{i}"], self[f"upper_{i}"], self[f"phys_{i}"]) = (
                        values[i * 3],
                        values[3 * i + 1],
                        values[3 * i + 2],
                    )
                (self.default,) = FLOAT64_u(block[-8:])

            elif conv == v4c.CONVERSION_TYPE_TABX:
                (
                    self.name_addr,
                    self.unit_addr,
                    self.comment_addr,
                    self.inv_conv_addr,
                ) = unpack_from("<4Q", block)

                links_nr = self.links_nr - 4

                links = unpack_from(f"<{links_nr}Q", block, 32)
                for i, link in enumerate(links[:-1]):
                    self[f"text_{i}"] = link
                self.default_addr = links[-1]

                (
                    self.conversion_type,
                    self.precision,
                    self.flags,
                    self.ref_param_nr,
                    self.val_param_nr,
                    self.min_phy_value,
                    self.max_phy_value,
                ) = unpack_from("<2B3H2d", block, 32 + links_nr * 8)

                values = unpack_from(f"<{links_nr - 1}d", block, 32 + links_nr * 8 + 24)
                for i, val in enumerate(values):
                    self[f"val_{i}"] = val

            elif conv == v4c.CONVERSION_TYPE_RTABX:
                (
                    self.name_addr,
                    self.unit_addr,
                    self.comment_addr,
                    self.inv_conv_addr,
                ) = unpack_from("<4Q", block)

                links_nr = self.links_nr - 4

                links = unpack_from(f"<{links_nr}Q", block, 32)
                for i, link in enumerate(links[:-1]):
                    self[f"text_{i}"] = link
                self.default_addr = links[-1]

                (
                    self.conversion_type,
                    self.precision,
                    self.flags,
                    self.ref_param_nr,
                    self.val_param_nr,
                    self.min_phy_value,
                    self.max_phy_value,
                ) = unpack_from("<2B3H2d", block, 32 + links_nr * 8)

                values = unpack_from(
                    "<{}d".format((links_nr - 1) * 2), block, 32 + links_nr * 8 + 24
                )
                self.default_lower = self.default_upper = 0
                for i in range(1, self.val_param_nr // 2):
                    j = 2 * i
                    self[f"lower_{i-1}"] = values[j]
                    self[f"upper_{i-1}"] = values[j + 1]

            elif conv == v4c.CONVERSION_TYPE_TTAB:
                (
                    self.name_addr,
                    self.unit_addr,
                    self.comment_addr,
                    self.inv_conv_addr,
                ) = unpack_from("<4Q", block)

                links_nr = self.links_nr - 4

                links = unpack_from(f"<{links_nr}Q", block, 32)
                for i, link in enumerate(links):
                    self[f"text_{i}"] = link

                (
                    self.conversion_type,
                    self.precision,
                    self.flags,
                    self.ref_param_nr,
                    self.val_param_nr,
                    self.min_phy_value,
                    self.max_phy_value,
                ) = unpack_from("<2B3H2d", block, 32 + links_nr * 8)

                values = unpack_from(
                    f"<{self.val_param_nr}d", block, 32 + links_nr * 8 + 24
                )
                for i, val in enumerate(values[:-1]):
                    self[f"val_{i}"] = val
                self.val_default = values[-1]

            elif conv == v4c.CONVERSION_TYPE_TRANS:
                (
                    self.name_addr,
                    self.unit_addr,
                    self.comment_addr,
                    self.inv_conv_addr,
                ) = unpack_from("<4Q", block)

                links_nr = self.links_nr - 4

                links = unpack_from(f"<{links_nr}Q", block, 32)

                for i in range((links_nr - 1) // 2):
                    j = 2 * i
                    self[f"input_{i}_addr"] = links[j]
                    self[f"output_{i}_addr"] = links[j + 1]
                self.default_addr = links[-1]

                (
                    self.conversion_type,
                    self.precision,
                    self.flags,
                    self.ref_param_nr,
                    self.val_param_nr,
                    self.min_phy_value,
                    self.max_phy_value,
                ) = unpack_from("<2B3H2d", block, 32 + links_nr * 8)

            self.name = get_text_v4(self.name_addr, stream, mapped=mapped)
            self.unit = get_text_v4(self.unit_addr, stream, mapped=mapped)
            self.comment = get_text_v4(self.comment_addr, stream, mapped=mapped)
            self.formula = ""
            self.referenced_blocks = None

            conv_type = conv

            if conv_type == v4c.CONVERSION_TYPE_ALG:
                self.formula = get_text_v4(self.formula_addr, stream, mapped=mapped)

            elif conv_type in v4c.TABULAR_CONVERSIONS:
                refs = self.referenced_blocks = {}
                if conv_type == v4c.CONVERSION_TYPE_TTAB:
                    tabs = self.links_nr - 4
                else:
                    tabs = self.links_nr - 4 - 1
                for i in range(tabs):
                    address = self[f"text_{i}"]
                    if address:
                        stream.seek(address)
                        _id = stream.read(4)

                        if _id == b'##TX':
                            block = TextBlock(address=address, stream=stream, mapped=mapped)
                            refs[f"text_{i}"] = block
                        elif _id == b'##CC':
                            block = ChannelConversion(address=address, stream=stream, mapped=mapped)
                            refs[f"text_{i}"] = block
                        else:
                            message = f'Expected "##TX" or "##CC" block @{hex(address)} but found "{_id}"'
                            logger.exception(message)
                            raise MdfException(message)

                    else:
                        refs[f"text_{i}"] = None
                if conv_type != v4c.CONVERSION_TYPE_TTAB:
                    address = self.default_addr
                    if address:
                        stream.seek(address)
                        _id = stream.read(4)

                        if _id == b'##TX':
                            block = TextBlock(address=address, stream=stream, mapped=mapped)
                            refs["default_addr"] = block
                        elif _id == b'##CC':
                            block = ChannelConversion(address=address, stream=stream, mapped=mapped)
                            refs["default_addr"] = block
                        else:
                            message = f'Expected "##TX" or "##CC" block @{hex(address)} but found "{_id}"'
                            logger.exception(message)
                            raise MdfException(message)
                    else:
                        refs["default_addr"] = None

            elif conv_type == v4c.CONVERSION_TYPE_TRANS:
                refs = self.referenced_blocks = {}
                # link_nr - common links (4) - default text link (1)
                for i in range((self.links_nr - 4 - 1) // 2):
                    for key in (f"input_{i}_addr", f"output_{i}_addr"):
                        address = self[key]
                        if address:
                            block = TextBlock(address=address, stream=stream, mapped=mapped)
                            refs[key] = block
                address = self.default_addr
                if address:
                    block = TextBlock(address=address, stream=stream, mapped=mapped)
                    refs["default_addr"] = block
                else:
                    refs["default_addr"] = None

        else:

            self.name = self.unit = self.comment = self.formula = ""
            self.referenced_blocks = None

            self.address = 0
            self.id = b"##CC"
            self.reserved0 = 0

            if kwargs["conversion_type"] == v4c.CONVERSION_TYPE_NON:
                self.block_len = v4c.CC_NONE_BLOCK_SIZE
                self.links_nr = 4
                self.name_addr = kwargs.get("name_addr", 0)
                self.unit_addr = kwargs.get("unit_addr", 0)
                self.comment_addr = kwargs.get("comment_addr", 0)
                self.inv_conv_addr = 0
                self.conversion_type = v4c.CONVERSION_TYPE_NON
                self.precision = 1
                self.flags = 0
                self.ref_param_nr = 0
                self.val_param_nr = 0
                self.min_phy_value = kwargs.get("min_phy_value", 0)
                self.max_phy_value = kwargs.get("max_phy_value", 0)

            elif kwargs["conversion_type"] == v4c.CONVERSION_TYPE_LIN:
                self.block_len = v4c.CC_LIN_BLOCK_SIZE
                self.links_nr = 4
                self.name_addr = kwargs.get("name_addr", 0)
                self.unit_addr = kwargs.get("unit_addr", 0)
                self.comment_addr = kwargs.get("comment_addr", 0)
                self.inv_conv_addr = kwargs.get("inv_conv_addr", 0)
                self.conversion_type = v4c.CONVERSION_TYPE_LIN
                self.precision = kwargs.get("precision", 1)
                self.flags = kwargs.get("flags", 0)
                self.ref_param_nr = 0
                self.val_param_nr = 2
                self.min_phy_value = kwargs.get("min_phy_value", 0)
                self.max_phy_value = kwargs.get("max_phy_value", 0)
                self.b = kwargs["b"]
                self.a = kwargs["a"]

            elif kwargs["conversion_type"] == v4c.CONVERSION_TYPE_ALG:
                self.block_len = v4c.CC_ALG_BLOCK_SIZE
                self.links_nr = 5
                self.name_addr = kwargs.get("name_addr", 0)
                self.unit_addr = kwargs.get("unit_addr", 0)
                self.comment_addr = kwargs.get("comment_addr", 0)
                self.inv_conv_addr = kwargs.get("inv_conv_addr", 0)
                self.formula_addr = kwargs.get("formula_addr", 0)
                self.conversion_type = v4c.CONVERSION_TYPE_ALG
                self.precision = kwargs.get("precision", 1)
                self.flags = kwargs.get("flags", 0)
                self.ref_param_nr = 1
                self.val_param_nr = 0
                self.min_phy_value = kwargs.get("min_phy_value", 0)
                self.max_phy_value = kwargs.get("max_phy_value", 0)
                self.formula = kwargs["formula"]

            elif kwargs["conversion_type"] in (
                v4c.CONVERSION_TYPE_TAB,
                v4c.CONVERSION_TYPE_TABI,
            ):

                nr = kwargs["val_param_nr"]

                self.block_len = 80 + 8 * nr
                self.links_nr = 4
                self.name_addr = kwargs.get("name_addr", 0)
                self.unit_addr = kwargs.get("unit_addr", 0)
                self.comment_addr = kwargs.get("comment_addr", 0)
                self.inv_conv_addr = kwargs.get("inv_conv_addr", 0)
                self.conversion_type = kwargs["conversion_type"]
                self.precision = kwargs.get("precision", 1)
                self.flags = kwargs.get("flags", 0)
                self.ref_param_nr = 0
                self.val_param_nr = nr
                self.min_phy_value = kwargs.get("min_phy_value", 0)
                self.max_phy_value = kwargs.get("max_phy_value", 0)

                for i in range(nr // 2):
                    self[f"raw_{i}"] = kwargs[f"raw_{i}"]
                    self[f"phys_{i}"] = kwargs[f"phys_{i}"]

            elif kwargs["conversion_type"] == v4c.CONVERSION_TYPE_RTAB:
                self.block_len = kwargs["val_param_nr"] * 8 + 80
                self.links_nr = 4
                self.name_addr = kwargs.get("name_addr", 0)
                self.unit_addr = kwargs.get("unit_addr", 0)
                self.comment_addr = kwargs.get("comment_addr", 0)
                self.inv_conv_addr = kwargs.get("inv_conv_addr", 0)
                self.conversion_type = v4c.CONVERSION_TYPE_RTAB
                self.precision = kwargs.get("precision", 0)
                self.flags = kwargs.get("flags", 0)
                self.ref_param_nr = 0
                self.val_param_nr = kwargs["val_param_nr"]
                self.min_phy_value = kwargs.get("min_phy_value", 0)
                self.max_phy_value = kwargs.get("max_phy_value", 0)
                for i in range((kwargs["val_param_nr"] - 1) // 3):
                    self[f"lower_{i}"] = kwargs[f"lower_{i}"]
                    self[f"upper_{i}"] = kwargs[f"upper_{i}"]
                    self[f"phys_{i}"] = kwargs[f"phys_{i}"]
                self.default = kwargs["default"]

            elif kwargs["conversion_type"] == v4c.CONVERSION_TYPE_RAT:

                self.block_len = 80 + 6 * 8
                self.links_nr = 4
                self.name_addr = kwargs.get("name_addr", 0)
                self.unit_addr = kwargs.get("unit_addr", 0)
                self.comment_addr = kwargs.get("comment_addr", 0)
                self.inv_conv_addr = kwargs.get("inv_conv_addr", 0)
                self.conversion_type = kwargs["conversion_type"]
                self.precision = kwargs.get("precision", 1)
                self.flags = kwargs.get("flags", 0)
                self.ref_param_nr = 0
                self.val_param_nr = kwargs.get("val_param_nr", 6)
                self.min_phy_value = kwargs.get("min_phy_value", 0)
                self.max_phy_value = kwargs.get("max_phy_value", 0)

                for i in range(1, 7):
                    self[f"P{i}"] = kwargs[f"P{i}"]

            elif kwargs["conversion_type"] == v4c.CONVERSION_TYPE_TABX:
                self.referenced_blocks = {}
                nr = kwargs["ref_param_nr"] - 1
                self.block_len = (nr * 8 * 2) + 88
                self.links_nr = nr + 5
                self.name_addr = kwargs.get("name_addr", 0)
                self.unit_addr = kwargs.get("unit_addr", 0)
                self.comment_addr = kwargs.get("comment_addr", 0)
                self.inv_conv_addr = kwargs.get("inv_conv_addr", 0)
                for i in range(nr):
                    key = f"text_{i}"
                    self[key] = 0
                    self.referenced_blocks[key] = TextBlock(text=kwargs[key])
                self.default_addr = 0
                key = "default_addr"
                if "default_addr" in kwargs:
                    default = kwargs["default_addr"]
                else:
                    default = kwargs.get("default", b"")
                if default:
                    self.referenced_blocks[key] = TextBlock(text=default)
                else:
                    self.referenced_blocks[key] = None
                self.conversion_type = v4c.CONVERSION_TYPE_TABX
                self.precision = kwargs.get("precision", 0)
                self.flags = kwargs.get("flags", 0)
                self.ref_param_nr = nr + 1
                self.val_param_nr = nr
                self.min_phy_value = kwargs.get("min_phy_value", 0)
                self.max_phy_value = kwargs.get("max_phy_value", 0)
                for i in range(nr):
                    self[f"val_{i}"] = kwargs[f"val_{i}"]

            elif kwargs["conversion_type"] == v4c.CONVERSION_TYPE_RTABX:
                self.referenced_blocks = {}
                nr = kwargs["ref_param_nr"] - 1
                self.block_len = (nr * 8 * 3) + 88
                self.links_nr = nr + 5
                self.name_addr = kwargs.get("name_addr", 0)
                self.unit_addr = kwargs.get("unit_addr", 0)
                self.comment_addr = kwargs.get("comment_addr", 0)
                self.inv_conv_addr = kwargs.get("inv_conv_addr", 0)
                for i in range(nr):
                    key = f"text_{i}"
                    self[key] = 0
                    self.referenced_blocks[key] = TextBlock(text=kwargs[key])
                self.default_addr = 0
                self.default_lower = self.default_upper = 0
                if "default_addr" in kwargs:
                    default = kwargs["default_addr"]
                else:
                    default = kwargs.get("default", b"")
                if default:
                    if b"{X}" in default:
                        default = (
                            default.decode("latin-1").replace("{X}", "X").split('"')[1]
                        )
                        default = ChannelConversion(
                            conversion_type=v4c.CONVERSION_TYPE_ALG, formula=default
                        )
                        self.referenced_blocks["default_addr"] = default
                    else:
                        self.referenced_blocks["default_addr"] = TextBlock(text=default)
                else:
                    self.referenced_blocks["default_addr"] = None
                self.conversion_type = v4c.CONVERSION_TYPE_RTABX
                self.precision = kwargs.get("precision", 0)
                self.flags = kwargs.get("flags", 0)
                self.ref_param_nr = nr + 1
                self.val_param_nr = nr * 2
                self.min_phy_value = kwargs.get("min_phy_value", 0)
                self.max_phy_value = kwargs.get("max_phy_value", 0)
                for i in range(nr):
                    self[f"lower_{i}"] = kwargs[f"lower_{i}"]
                    self[f"upper_{i}"] = kwargs[f"upper_{i}"]

            elif kwargs["conversion_type"] == v4c.CONVERSION_TYPE_TTAB:
                self.block_len = ((kwargs["links_nr"] - 4) * 8 * 2) + 88
                self.links_nr = kwargs["links_nr"]
                self.name_addr = kwargs.get("name_addr", 0)
                self.unit_addr = kwargs.get("unit_addr", 0)
                self.comment_addr = kwargs.get("comment_addr", 0)
                self.inv_conv_addr = kwargs.get("inv_conv_addr", 0)
                for i in range(kwargs["links_nr"] - 4):
                    self[f"text_{i}"] = kwargs.get(f"text_{i}", 0)
                self.conversion_type = v4c.CONVERSION_TYPE_TTAB
                self.precision = kwargs.get("precision", 0)
                self.flags = kwargs.get("flags", 0)
                self.ref_param_nr = kwargs["links_nr"] - 4
                self.val_param_nr = kwargs["links_nr"] - 4 + 1
                self.min_phy_value = kwargs.get("min_phy_value", 0)
                self.max_phy_value = kwargs.get("max_phy_value", 0)
                for i in range(kwargs["links_nr"] - 4):
                    self[f"val_{i}"] = kwargs[f"val_{i}"]
                self.val_default = kwargs["val_default"]

            else:
                message = "Conversion {} dynamic creation not implementated"
                message = message.format(kwargs["conversion_type"])
                logger.exception(message)
                raise MdfException(message)

    def to_blocks(self, address, blocks, defined_texts, cc_map):
        text = self.name
        if text:
            if text in defined_texts:
                self.name_addr = defined_texts[text]
            else:
                tx_block = TextBlock(text=text)
                self.name_addr = address
                defined_texts[text] = address
                tx_block.address = address
                address += tx_block.block_len
                blocks.append(tx_block)
        else:
            self.name_addr = 0

        text = self.unit
        if text:
            if text in defined_texts:
                self.unit_addr = defined_texts[text]
            else:
                tx_block = TextBlock(text=text)
                self.unit_addr = address
                defined_texts[text] = address
                tx_block.address = address
                address += tx_block.block_len
                blocks.append(tx_block)
        else:
            self.unit_addr = 0

        if self.conversion_type == v4c.CONVERSION_TYPE_ALG:
            text = self.formula
            if text:
                if text in defined_texts:
                    self.formula_addr = defined_texts[text]
                else:
                    tx_block = TextBlock(text=text)
                    self.formula_addr = address
                    defined_texts[text] = address
                    tx_block.address = address
                    address += tx_block.block_len
                    blocks.append(tx_block)
            else:
                self.formula_addr = 0

        text = self.comment
        if text:
            if text in defined_texts:
                self.comment_addr = defined_texts[text]
            else:
                meta = text.startswith("<CCcomment")
                tx_block = TextBlock(text=text, meta=meta)
                self.comment_addr = address
                defined_texts[text] = address
                tx_block.address = address
                address += tx_block.block_len
                blocks.append(tx_block)
        else:
            self.comment_addr = 0

        if self.referenced_blocks:
            for key, block in self.referenced_blocks.items():
                if block:
                    if block["id"] == b"##TX":
                        text = block["text"]
                        if text in defined_texts:
                            self[key] = defined_texts[text]
                        else:
                            defined_texts[text] = address
                            blocks.append(block)
                            self[key] = address
                            address += block["block_len"]
                    else:
                        address = block.to_blocks(
                            address, blocks, defined_texts, cc_map
                        )
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
            address += self.block_len

        return address

    def convert(self, values):
        conversion_type = self.conversion_type
        if conversion_type == v4c.CONVERSION_TYPE_NON:
            pass
        elif conversion_type == v4c.CONVERSION_TYPE_LIN:
            a = self.a
            b = self.b
            if (a, b) != (1, 0):
                values = values * a
                if b:
                    values += b
        elif conversion_type == v4c.CONVERSION_TYPE_RAT:
            P1 = self.P1
            P2 = self.P2
            P3 = self.P3
            P4 = self.P4
            P5 = self.P5
            P6 = self.P6

            X = values
            if (P1, P4, P5, P6) == (0, 0, 0, 1):
                if (P2, P3) != (1, 0):
                    values = values * P2
                    if P3:
                        values += P3
            elif (P3, P4, P5, P6) == (0, 0, 1, 0):
                if (P1, P2) != (1, 0):
                    values = values * P1
                    if P2:
                        values += P2
            else:
                try:
                    values = evaluate(v4c.CONV_RAT_TEXT)
                except TypeError:
                    values = (P1 * X ** 2 + P2 * X + P3) / (P4 * X ** 2 + P5 * X + P6)

        elif conversion_type == v4c.CONVERSION_TYPE_ALG:
            X = values
            values = evaluate(self.formula)

        elif conversion_type in (v4c.CONVERSION_TYPE_TABI, v4c.CONVERSION_TYPE_TAB):
            nr = self.val_param_nr // 2
            raw_vals = np.array([self[f"raw_{i}"] for i in range(nr)])
            phys = np.array([self[f"phys_{i}"] for i in range(nr)])

            if conversion_type == v4c.CONVERSION_TYPE_TABI:
                values = np.interp(values, raw_vals, phys)
            else:
                idx = np.searchsorted(raw_vals, values)
                idx = np.clip(idx, 0, len(raw_vals) - 1)
                values = phys[idx]

        elif conversion_type == v4c.CONVERSION_TYPE_RTAB:
            nr = (self.val_param_nr - 1) // 3
            lower = np.array([self[f"lower_{i}"] for i in range(nr)])
            upper = np.array([self[f"upper_{i}"] for i in range(nr)])
            phys = np.array([self[f"phys_{i}"] for i in range(nr)])
            default = self.default

            if values.dtype.kind == "f":
                idx1 = np.searchsorted(lower, values, side="right") - 1
                idx2 = np.searchsorted(upper, values, side="right")
            else:
                idx1 = np.searchsorted(lower, values, side="right") - 1
                idx2 = np.searchsorted(upper, values, side="right") - 1

            idx_ne = np.nonzero(idx1 != idx2)[0]
            idx_eq = np.nonzero(idx1 == idx2)[0]

            new_values = np.zeros(len(values), dtype=phys.dtype)

            if len(idx_ne):
                new_values[idx_ne] = default
            if len(idx_eq):
                new_values[idx_eq] = phys[idx1[idx_eq]]

            values = new_values

        elif conversion_type == v4c.CONVERSION_TYPE_TABX:
            nr = self.val_param_nr
            raw_vals = np.array([self[f"val_{i}"] for i in range(nr)])

            phys = []
            for i in range(nr):
                try:
                    value = self.referenced_blocks[f"text_{i}"].text
                except AttributeError:
                    value = self.referenced_blocks[f"text_{i}"]
                except TypeError:
                    value = b""
                phys.append(value)

            default = self.referenced_blocks.get("default_addr", {})
            if default is None:
                default = b""
            else:
                try:
                    default = default.text
                except AttributeError:
                    pass
                except TypeError:
                    default = b""

            phys.insert(0, default)
            raw_vals = np.insert(raw_vals, 0, raw_vals[0] - 1)
            indexes = np.searchsorted(raw_vals, values)
            np.place(indexes, indexes >= len(raw_vals), 0)

            all_values = list(phys) + [default]

            if all(isinstance(val, bytes) for val in all_values):
                phys = np.array(phys)
                values = phys[indexes]
            else:
                new_values = []
                for i, idx in enumerate(indexes):
                    item = phys[idx]

                    if isinstance(item, bytes):
                        new_values.append(item)
                    else:
                        new_values.append(item.convert(values[i : i + 1])[0])

                if all(isinstance(v, bytes) for v in new_values):
                    values = np.array(new_values)
                else:
                    values = np.array(
                        [np.nan if isinstance(v, bytes) else v for v in new_values]
                    )

        elif conversion_type == v4c.CONVERSION_TYPE_RTABX:
            nr = self.val_param_nr // 2 - 1

            phys = []
            for i in range(nr):
                try:
                    value = self.referenced_blocks[f"text_{i}"].text
                except AttributeError:
                    value = self.referenced_blocks[f"text_{i}"]
                except TypeError:
                    value = b""
                phys.append(value)

            default = self.referenced_blocks.get("default_addr", {})
            if default is None:
                default = b""
            else:
                try:
                    default = default.text
                except AttributeError:
                    pass
                except TypeError:
                    default = b""

            lower = np.array([self[f"lower_{i}"] for i in range(nr)])
            upper = np.array([self[f"upper_{i}"] for i in range(nr)])

            all_values = phys + [default]

            idx1 = np.searchsorted(lower, values, side="right") - 1
            idx2 = np.searchsorted(upper, values, side="left")

            idx_ne = np.nonzero(idx1 != idx2)[0]
            idx_eq = np.nonzero(idx1 == idx2)[0]

            if all(isinstance(val, bytes) for val in all_values):
                phys = np.array(phys)
                all_values = np.array(all_values)

                new_values = np.zeros(len(values), dtype=all_values.dtype)

                if len(idx_ne):
                    new_values[idx_ne] = default
                if len(idx_eq):
                    new_values[idx_eq] = phys[idx1[idx_eq]]

                values = new_values
            else:
                new_values = []
                for i, val in enumerate(values):
                    if i in idx_ne:
                        item = default
                    else:
                        item = phys[idx1[i]]

                    if isinstance(item, bytes):
                        new_values.append(item)
                    else:
                        try:
                            new_values.append(item.convert(values[i : i + 1])[0])
                        except:
                            print(item, default, phys)
                            1/0

                if all(isinstance(v, bytes) for v in new_values):
                    values = np.array(new_values)
                else:
                    values = np.array(
                        [np.nan if isinstance(v, bytes) else v for v in new_values]
                    )

        elif conversion_type == v4c.CONVERSION_TYPE_TTAB:
            nr = self.val_param_nr - 1

            raw_values = [
                self.referenced_blocks[f"text_{i}"].text.strip(b"\0") for i in range(nr)
            ]
            phys = [self[f"val_{i}"] for i in range(nr)]
            default = self.val_default

            new_values = []
            for val in values:
                try:
                    val = phys[raw_values.index(val)]
                except ValueError:
                    val = default
                new_values.append(val)

            values = np.array(new_values)

        elif conversion_type == v4c.CONVERSION_TYPE_TRANS:
            nr = (self.ref_param_nr - 1) // 2

            in_ = [
                self.referenced_blocks[f"input_{i}_addr"].text.strip(b"\0")
                for i in range(nr)
            ]

            out_ = [
                self.referenced_blocks[f"output_{i}_addr"].text.strip(b"\0")
                for i in range(nr)
            ]
            default = self.referenced_blocks["default_addr"].text.strip(b"\0")

            new_values = []
            for val in values:
                try:
                    val = out_[in_.index(val.strip(b"\0"))]
                except ValueError:
                    val = default
                new_values.append(val)

            values = np.array(new_values)

        return values

    def metadata(self, indent=""):
        if self.conversion_type == v4c.CONVERSION_TYPE_NON:
            keys = v4c.KEYS_CONVERSION_NONE
        elif self.conversion_type == v4c.CONVERSION_TYPE_LIN:
            keys = v4c.KEYS_CONVERSION_LINEAR
        elif self.conversion_type == v4c.CONVERSION_TYPE_RAT:
            keys = v4c.KEYS_CONVERSION_RAT
        elif self.conversion_type == v4c.CONVERSION_TYPE_ALG:
            keys = v4c.KEYS_CONVERSION_ALGEBRAIC
        elif self.conversion_type in (
            v4c.CONVERSION_TYPE_TABI,
            v4c.CONVERSION_TYPE_TAB,
        ):
            keys = v4c.KEYS_CONVERSION_NONE
            for i in range(self.val_param_nr // 2):
                keys += (f"raw_{i}", f"phys_{i}")
        elif self.conversion_type == v4c.CONVERSION_TYPE_RTAB:
            keys = v4c.KEYS_CONVERSION_NONE
            for i in range(self.val_param_nr // 3):
                keys += (f"lower_{i}", f"upper_{i}", f"phys_{i}")
            keys += ("default",)
        elif self.conversion_type == v4c.CONVERSION_TYPE_TABX:
            keys = (
                "id",
                "reserved0",
                "block_len",
                "links_nr",
                "name_addr",
                "unit_addr",
                "comment_addr",
                "inv_conv_addr",
            )
            keys += tuple(f"text_{i}" for i in range(self.links_nr - 4 - 1))
            keys += ("default_addr",)
            keys += (
                "conversion_type",
                "precision",
                "flags",
                "ref_param_nr",
                "val_param_nr",
                "min_phy_value",
                "max_phy_value",
            )
            keys += tuple(f"val_{i}" for i in range(self.val_param_nr))
        elif self.conversion_type == v4c.CONVERSION_TYPE_RTABX:
            keys = (
                "id",
                "reserved0",
                "block_len",
                "links_nr",
                "name_addr",
                "unit_addr",
                "comment_addr",
                "inv_conv_addr",
            )
            keys += tuple(f"text_{i}" for i in range(self.links_nr - 4 - 1))
            keys += ("default_addr",)
            keys += (
                "conversion_type",
                "precision",
                "flags",
                "ref_param_nr",
                "val_param_nr",
                "min_phy_value",
                "max_phy_value",
            )
            for i in range(self.val_param_nr // 2):
                keys += (f"lower_{i}", f"upper_{i}")
        elif self.conversion_type == v4c.CONVERSION_TYPE_TTAB:
            keys = (
                "id",
                "reserved0",
                "block_len",
                "links_nr",
                "name_addr",
                "unit_addr",
                "comment_addr",
                "inv_conv_addr",
            )
            keys += tuple(f"text_{i}" for i in range(self.links_nr - 4))
            keys += (
                "conversion_type",
                "precision",
                "flags",
                "ref_param_nr",
                "val_param_nr",
                "min_phy_value",
                "max_phy_value",
            )
            keys += tuple(f"val_{i}" for i in range(self.val_param_nr - 1))
            keys += ("val_default",)
        elif self.conversion_type == v4c.CONVERSION_TYPE_TRANS:
            keys = (
                "id",
                "reserved0",
                "block_len",
                "links_nr",
                "name_addr",
                "unit_addr",
                "comment_addr",
                "inv_conv_addr",
            )
            for i in range((self.links_nr - 4 - 1) // 2):
                keys += (f"input_{i}_addr", f"output_{i}_addr")
            keys += (
                "default_addr",
                "conversion_type",
                "precision",
                "flags",
                "ref_param_nr",
                "val_param_nr",
                "min_phy_value",
                "max_phy_value",
            )
            keys += tuple(f"val_{i}" for i in range(self.val_param_nr - 1))
        max_len = max(len(key) for key in keys)
        template = f"{{: <{max_len}}}: {{}}"

        metadata = []
        lines = f"""
name: {self.name}
unit: {self.unit}
address: {hex(self.address)}
comment: {self.comment}
formula: {self.formula}

""".split("\n")
        for key in keys:
            val = getattr(self, key)
            if key.endswith("addr") or key.startswith("text_"):
                lines.append(template.format(key, hex(val)))
            elif isinstance(val, float):
                lines.append(template.format(key, round(val, 6)))
            else:
                if isinstance(val, bytes):
                    lines.append(template.format(key, val.strip(b"\0")))
                else:
                    lines.append(template.format(key, val))

        if self.referenced_blocks:
            max_len = max(len(key) for key in self.referenced_blocks)
            template = f"{{: <{max_len}}}: {{}}"

            lines.append("")
            lines.append("Referenced blocks:")
            for key, block in self.referenced_blocks.items():
                if isinstance(block, TextBlock):
                    lines.append(template.format(key, block["text"].strip(b"\0")))
                else:
                    lines.append(template.format(key, ""))
                    lines.extend(block.metadata(indent + "    ").split("\n"))

        for line in lines:
            if not line:
                metadata.append(line)
            else:
                for wrapped_line in wrap(
                    line, initial_indent=indent, subsequent_indent=indent, width=120
                ):
                    metadata.append(wrapped_line)

        return "\n".join(metadata)

    def __getitem__(self, item):
        try:
            return self.__getattribute__(item)
        except:
            print(self)
            raise

    def __setitem__(self, item, value):
        self.__setattr__(item, value)

    def __contains__(self, item):
        return hasattr(self, item)

    def __bytes__(self):
        if self.conversion_type == v4c.CONVERSION_TYPE_NON:
            result = v4c.CONVERSION_NONE_PACK(
                self.id,
                self.reserved0,
                self.block_len,
                self.links_nr,
                self.name_addr,
                self.unit_addr,
                self.comment_addr,
                self.inv_conv_addr,
                self.conversion_type,
                self.precision,
                self.flags,
                self.ref_param_nr,
                self.val_param_nr,
                self.min_phy_value,
                self.max_phy_value,
            )
        elif self.conversion_type == v4c.CONVERSION_TYPE_LIN:
            result = v4c.CONVERSION_LINEAR_PACK(
                self.id,
                self.reserved0,
                self.block_len,
                self.links_nr,
                self.name_addr,
                self.unit_addr,
                self.comment_addr,
                self.inv_conv_addr,
                self.conversion_type,
                self.precision,
                self.flags,
                self.ref_param_nr,
                self.val_param_nr,
                self.min_phy_value,
                self.max_phy_value,
                self.b,
                self.a,
            )
        elif self.conversion_type == v4c.CONVERSION_TYPE_RAT:
            result = v4c.CONVERSION_RAT_PACK(
                self.id,
                self.reserved0,
                self.block_len,
                self.links_nr,
                self.name_addr,
                self.unit_addr,
                self.comment_addr,
                self.inv_conv_addr,
                self.conversion_type,
                self.precision,
                self.flags,
                self.ref_param_nr,
                self.val_param_nr,
                self.min_phy_value,
                self.max_phy_value,
                self.P1,
                self.P2,
                self.P3,
                self.P4,
                self.P5,
                self.P6,
            )
        elif self.conversion_type == v4c.CONVERSION_TYPE_ALG:
            result = v4c.CONVERSION_ALGEBRAIC_PACK(
                self.id,
                self.reserved0,
                self.block_len,
                self.links_nr,
                self.name_addr,
                self.unit_addr,
                self.comment_addr,
                self.inv_conv_addr,
                self.formula_addr,
                self.conversion_type,
                self.precision,
                self.flags,
                self.ref_param_nr,
                self.val_param_nr,
                self.min_phy_value,
                self.max_phy_value,
            )
        elif self.conversion_type in (
            v4c.CONVERSION_TYPE_TABI,
            v4c.CONVERSION_TYPE_TAB,
        ):
            fmt = "<4sI{}Q2B3H{}d".format(self.links_nr + 2, self.val_param_nr + 2)
            keys = v4c.KEYS_CONVERSION_NONE
            for i in range(self.val_param_nr // 2):
                keys += (f"raw_{i}", f"phys_{i}")
            result = pack(fmt, *[getattr(self, key) for key in keys])
        elif self.conversion_type == v4c.CONVERSION_TYPE_RTAB:
            fmt = "<4sI{}Q2B3H{}d".format(self.links_nr + 2, self.val_param_nr + 2)
            keys = v4c.KEYS_CONVERSION_NONE
            for i in range(self.val_param_nr // 3):
                keys += (f"lower_{i}", f"upper_{i}", f"phys_{i}")
            keys += ("default",)
            result = pack(fmt, *[getattr(self, key) for key in keys])
        elif self.conversion_type == v4c.CONVERSION_TYPE_TABX:
            fmt = "<4sI{}Q2B3H{}d".format(self.links_nr + 2, self.val_param_nr + 2)
            keys = (
                "id",
                "reserved0",
                "block_len",
                "links_nr",
                "name_addr",
                "unit_addr",
                "comment_addr",
                "inv_conv_addr",
            )
            keys += tuple(f"text_{i}" for i in range(self.links_nr - 4 - 1))
            keys += ("default_addr",)
            keys += (
                "conversion_type",
                "precision",
                "flags",
                "ref_param_nr",
                "val_param_nr",
                "min_phy_value",
                "max_phy_value",
            )
            keys += tuple(f"val_{i}" for i in range(self.val_param_nr))
            result = pack(fmt, *[getattr(self, key) for key in keys])
        elif self.conversion_type == v4c.CONVERSION_TYPE_RTABX:
            fmt = "<4sI{}Q2B3H{}d".format(self.links_nr + 2, self.val_param_nr + 2)
            keys = (
                "id",
                "reserved0",
                "block_len",
                "links_nr",
                "name_addr",
                "unit_addr",
                "comment_addr",
                "inv_conv_addr",
            )
            keys += tuple(f"text_{i}" for i in range(self.links_nr - 4 - 1))
            keys += ("default_addr",)
            keys += (
                "conversion_type",
                "precision",
                "flags",
                "ref_param_nr",
                "val_param_nr",
                "min_phy_value",
                "max_phy_value",
                "default_lower",
                "default_upper",
            )
            for i in range(self.val_param_nr // 2 - 1):
                keys += (f"lower_{i}", f"upper_{i}")
            result = pack(fmt, *[getattr(self, key) for key in keys])
        elif self.conversion_type == v4c.CONVERSION_TYPE_TTAB:
            fmt = "<4sI{}Q2B3H{}d".format(self.links_nr + 2, self.val_param_nr + 2)
            keys = (
                "id",
                "reserved0",
                "block_len",
                "links_nr",
                "name_addr",
                "unit_addr",
                "comment_addr",
                "inv_conv_addr",
            )
            keys += tuple(f"text_{i}" for i in range(self.links_nr - 4))
            keys += (
                "conversion_type",
                "precision",
                "flags",
                "ref_param_nr",
                "val_param_nr",
                "min_phy_value",
                "max_phy_value",
            )
            keys += tuple(f"val_{i}" for i in range(self.val_param_nr - 1))
            keys += ("val_default",)
            result = pack(fmt, *[getattr(self, key) for key in keys])
        elif self.conversion_type == v4c.CONVERSION_TYPE_TRANS:
            fmt = "<4sI{}Q2B3H{}d".format(self.links_nr + 2, self.val_param_nr + 2)
            keys = (
                "id",
                "reserved0",
                "block_len",
                "links_nr",
                "name_addr",
                "unit_addr",
                "comment_addr",
                "inv_conv_addr",
            )
            for i in range((self.links_nr - 4 - 1) // 2):
                keys += (f"input_{i}_addr", f"output_{i}_addr")
            keys += (
                "default_addr",
                "conversion_type",
                "precision",
                "flags",
                "ref_param_nr",
                "val_param_nr",
                "min_phy_value",
                "max_phy_value",
            )
            keys += tuple(f"val_{i}" for i in range(self.val_param_nr - 1))

            result = pack(fmt, *[getattr(self, key) for key in keys])
        return result

    def __str__(self):
        return "<ChannelConversion (name: {}, unit: {}, comment: {}, formula: {}, referenced blocks: {}, address: {}, fields: {})>".format(
            self.name,
            self.unit,
            self.comment,
            self.formula,
            self.referenced_blocks,
            self.address,
            block_fields(self),
        )


class DataBlock:
    """Common implementation for DTBLOCK/RDBLOCK/SDBLOCK

    *DataBlock* has the following attributes, that are also available as
    dict like key-value pairs

    DTBLOCK fields

    * ``id`` - bytes : block ID; b'##DT' for DTBLOCK, b'##RD' for RDBLOCK or
      b'##SD' for SDBLOCK
    * ``reserved0`` - int : reserved bytes
    * ``block_len`` - int : block bytes size
    * ``links_nr`` - int : number of links
    * ``data`` - bytes : raw samples

    Other attributes

    * ``address`` - int : data block address

    Parameters
    ----------
    address : int
        DTBLOCK/RDBLOCK/SDBLOCK address inside the file
    stream : int
        file handle
    reduction : bool
        sample reduction data block

    """

    __slots__ = ("address", "id", "reserved0", "block_len", "links_nr", "data")

    def __init__(self, **kwargs):

        try:
            self.address = address = kwargs["address"]
            stream = kwargs["stream"]
            mapped = kwargs.get("mapped", False)

            if mapped:
                (self.id, self.reserved0, self.block_len, self.links_nr) = COMMON_uf(
                    stream, address
                )
                address += COMMON_SIZE
                if self.id not in (b"##DT", b"##RD", b"##SD"):
                    message = f'Expected "##DT", "##RD" or "##SD" block @{hex(address)} but found "{self.id}"'
                    logger.exception(message)
                    raise MdfException(message)
                self.data = stream[address: address + self.block_len]
            else:

                stream.seek(address)

                (self.id, self.reserved0, self.block_len, self.links_nr) = COMMON_u(
                    stream.read(COMMON_SIZE)
                )

                if self.id not in (b"##DT", b"##RD", b"##SD"):
                    message = f'Expected "##DT", "##RD" or "##SD" block @{hex(address)} but found "{self.id}"'
                    logger.exception(message)
                    raise MdfException(message)

                self.data = stream.read(self.block_len - COMMON_SIZE)

        except KeyError:
            self.address = 0
            type = kwargs.get("type", "DT")
            if type not in ("DT", "SD", "RD"):
                type = "DT"

            self.id = "##{}".format(type).encode("ascii")
            self.reserved0 = 0
            self.block_len = len(kwargs["data"]) + COMMON_SIZE
            self.links_nr = 0
            self.data = kwargs["data"]

    def __getitem__(self, item):
        return self.__getattribute__(item)

    def __setitem__(self, item, value):
        self.__setattr__(item, value)

    def __bytes__(self):
        return v4c.COMMON_p(
            self.id, self.reserved0, self.block_len, self.links_nr
        ) + self.data


class DataZippedBlock(object):
    """*DataZippedBlock* has the following attributes, that are also available
    as dict like key-value pairs

    DZBLOCK fields

    * ``id`` - bytes : block ID; always b'##DZ'
    * ``reserved0`` - int : reserved bytes
    * ``block_len`` - int : block bytes size
    * ``links_nr`` - int : number of links
    * ``original_type`` - bytes : b'DT' or b'SD'
    * ``zip_type`` - int : zip algorithm used
    * ``reserved1`` - int : reserved bytes
    * ``param`` - int : for transpose deflate the record size used for
      transposition
    * ``original_size`` - int : size of the original uncompressed raw bytes
    * ``zip_size`` - int : size of compressed bytes
    * ``data`` - bytes : compressed bytes

    Other attributes

    * ``address`` - int : data zipped block address
    * ``return_unzipped`` - bool : decompress data when accessing the 'data'
      key

    Parameters
    ----------
    address : int
        DTBLOCK address inside the file
    stream : int
        file handle

    """

    __slots__ = (
        "address",
        "_prevent_data_setitem",
        "return_unzipped",
        "id",
        "reserved0",
        "block_len",
        "links_nr",
        "original_type",
        "zip_type",
        "reserved1",
        "param",
        "original_size",
        "zip_size",
        "data",
    )

    def __init__(self, **kwargs):

        self._prevent_data_setitem = True
        try:
            self.address = address = kwargs["address"]
            stream = kwargs["stream"]
            stream.seek(address)

            (
                self.id,
                self.reserved0,
                self.block_len,
                self.links_nr,
                self.original_type,
                self.zip_type,
                self.reserved1,
                self.param,
                self.original_size,
                self.zip_size,
            ) = unpack(v4c.FMT_DZ_COMMON, stream.read(v4c.DZ_COMMON_SIZE))

            if self.id != b"##DZ":
                message = f'Expected "##DZ" block @{hex(address)} but found "{self.id}"'

                logger.exception(message)
                raise MdfException(message)

            self.data = stream.read(self.zip_size)

        except KeyError:
            self._prevent_data_setitem = False
            self.address = 0

            data = kwargs["data"]

            self.id = b"##DZ"
            self.reserved0 = 0
            self.block_len = 0
            self.links_nr = 0
            self.original_type = kwargs.get("original_type", b"DT")
            self.zip_type = kwargs.get("zip_type", v4c.FLAG_DZ_DEFLATE)
            self.reserved1 = 0
            if self.zip_type == v4c.FLAG_DZ_DEFLATE:
                self.param = 0
            else:
                self.param = kwargs["param"]

            # since prevent_data_setitem is False the rest of the keys will be
            # handled by __setitem__
            self.data = data

        self._prevent_data_setitem = False
        self.return_unzipped = True

    def __setattr__(self, item, value):
        if item == "data" and not self._prevent_data_setitem:
            data = value
            original_size = len(data)
            self.original_size = original_size

            if self.zip_type == v4c.FLAG_DZ_DEFLATE:
                data = compress(data, 1)
            else:
                cols = self.param
                lines = original_size // cols

                if lines * cols < original_size:
                    data = (
                        np.fromstring(data[: lines * cols], dtype=np.uint8)
                        .reshape((lines, cols))
                        .T.tostring()
                    ) + data[lines * cols :]

                else:
                    data = (
                        np.fromstring(data, dtype=np.uint8)
                        .reshape((lines, cols))
                        .T.tostring()
                    )
                data = compress(data, 1)

            zipped_size = len(data)
            self.zip_size = zipped_size
            self.block_len = zipped_size + v4c.DZ_COMMON_SIZE
            DataZippedBlock.__dict__[item].__set__(self, data)
        else:
            DataZippedBlock.__dict__[item].__set__(self, value)

    def __getattribute__(self, item):
        if item == "data":
            if self.return_unzipped:
                data = DataZippedBlock.__dict__[item].__get__(self)
                original_size = self.original_size
                data = decompress(data, 0, original_size)
                if self.zip_type == v4c.FLAG_DZ_TRANPOSED_DEFLATE:
                    cols = self.param
                    lines = original_size // cols

                    if lines * cols < original_size:
                        data = (
                            np.fromstring(data[: lines * cols], dtype=np.uint8)
                            .reshape((cols, lines))
                            .T.tostring()
                        ) + data[lines * cols :]
                    else:
                        data = (
                            np.fromstring(data, dtype=np.uint8)
                            .reshape((cols, lines))
                            .T.tostring()
                        )
            else:
                data = DataZippedBlock.__dict__[item].__get__(self)
            value = data
        else:
            value = DataZippedBlock.__dict__[item].__get__(self)
        return value

    def __setitem__(self, item, value):
        self.__setattr__(item, value)

    def __getitem__(self, item):
        return self.__getattribute__(item)

    def __bytes__(self):
        self.return_unzipped = False
        data = v4c.DZ_COMMON_p(
            self.id,
            self.reserved0,
            self.block_len,
            self.links_nr,
            self.original_type,
            self.zip_type,
            self.reserved1,
            self.param,
            self.original_size,
            self.zip_size,
        ) + self.data
        self.return_unzipped = True
        return data


class DataGroup:
    """
    *DataGroup* has the following attributes, that are also available as
    dict like key-value pairs

    DGBLOCK fields

    * ``id`` - bytes : block ID; always b'##DG'
    * ``reserved0`` - int : reserved bytes
    * ``block_len`` - int : block bytes size
    * ``links_nr`` - int : number of links
    * ``next_dg_addr`` - int : address of next data group block
    * ``first_cg_addr`` - int : address of first channel group for this data
      group
    * ``data_block_addr`` - int : address of DTBLOCK, DZBLOCK, DLBLOCK or
      HLBLOCK that contains the raw samples for this data group
    * ``comment_addr`` - int : address of TXBLOCK/MDBLOCK tha contains the
      data group comment
    * ``record_id_len`` - int : size of record ID used in case of unsorted
      data groups; can be 1, 2, 4 or 8
    * ``reserved1`` - int : reserved bytes

    Other attributes

    * ``address`` - int : dat group address
    * ``comment`` - str : data group comment

    """

    __slots__ = (
        "address",
        "comment",
        "id",
        "reserved0",
        "block_len",
        "links_nr",
        "next_dg_addr",
        "first_cg_addr",
        "data_block_addr",
        "comment_addr",
        "record_id_len",
        "reserved1",
    )

    def __init__(self, **kwargs):

        self.comment = ""

        try:
            self.address = address = kwargs["address"]
            stream = kwargs["stream"]
            mapped = kwargs.get("mapped", False)

            if mapped:
                stream.seek(address)

                (
                    self.id,
                    self.reserved0,
                    self.block_len,
                    self.links_nr,
                    self.next_dg_addr,
                    self.first_cg_addr,
                    self.data_block_addr,
                    self.comment_addr,
                    self.record_id_len,
                    self.reserved1,
                ) = v4c.DATA_GROUP_uf(stream, address)
            else:

                stream.seek(address)

                (
                    self.id,
                    self.reserved0,
                    self.block_len,
                    self.links_nr,
                    self.next_dg_addr,
                    self.first_cg_addr,
                    self.data_block_addr,
                    self.comment_addr,
                    self.record_id_len,
                    self.reserved1,
                ) = v4c.DATA_GROUP_u(stream.read(v4c.DG_BLOCK_SIZE))

            if self.id != b"##DG":
                message = f'Expected "##DG" block @{hex(address)} but found "{self.id}"'

                logger.exception(message)
                raise MdfException(message)

            self.comment = get_text_v4(self.comment_addr, stream, mapped=mapped)

        except KeyError:

            self.address = 0
            self.id = b"##DG"
            self.reserved0 = kwargs.get("reserved0", 0)
            self.block_len = kwargs.get("block_len", v4c.DG_BLOCK_SIZE)
            self.links_nr = kwargs.get("links_nr", 4)
            self.next_dg_addr = kwargs.get("next_dg_addr", 0)
            self.first_cg_addr = kwargs.get("first_cg_addr", 0)
            self.data_block_addr = kwargs.get("data_block_addr", 0)
            self.comment_addr = kwargs.get("comment_addr", 0)
            self.record_id_len = kwargs.get("record_id_len", 0)
            self.reserved1 = kwargs.get("reserved1", b"\00" * 7)

    def copy(self):
        dg = DataGroup(
            id=self.id,
            reserved0=self.reserved0,
            block_len=self.block_len,
            links_nr=self.links_nr,
            next_dg_addr=self.next_dg_addr,
            first_cg_addr=self.first_cg_addr,
            data_block_addr=self.data_block_addr,
            comment_addr=self.comment_addr,
            record_id_len=self.record_id_len,
            reserved1=self.reserved1,
        )
        dg.comment = self.comment
        dg.address = self.address

        return dg

    def to_blocks(self, address, blocks, defined_texts):
        text = self.comment
        if text:
            if text in defined_texts:
                self.comment_addr = defined_texts[text]
            else:
                meta = text.startswith("<DGcomment")
                tx_block = TextBlock(text=text, meta=meta)
                self.comment_addr = address
                defined_texts[text] = address
                tx_block.address = address
                address += tx_block.block_len
                blocks.append(tx_block)
        else:
            self.comment_addr = 0

        blocks.append(self)
        self.address = address
        address += self.block_len

        return address

    def __getitem__(self, item):
        return self.__getattribute__(item)

    def __setitem__(self, item, value):
        self.__setattr__(item, value)

    def __bytes__(self):
        result = v4c.DATA_GROUP_p(
            self.id,
            self.reserved0,
            self.block_len,
            self.links_nr,
            self.next_dg_addr,
            self.first_cg_addr,
            self.data_block_addr,
            self.comment_addr,
            self.record_id_len,
            self.reserved1,
        )
        return result


class _DataListBase:
    __slots__ = (
        "address",
        "id",
        "reserved0",
        "block_len",
        "links_nr",
        "next_dl_addr",
        "flags",
        "reserved1",
        "data_block_nr",
        "data_block_len",
    )


class DataList(_DataListBase):
    """
    *DataList* has the following attributes, that are also available as
    dict like key-value pairs

    DLBLOCK common fields

    * ``id`` - bytes : block ID; always b'##DL'
    * ``reserved0`` - int : reserved bytes
    * ``block_len`` - int : block bytes size
    * ``links_nr`` - int : number of links
    * ``next_dl_addr`` - int : address of next DLBLOCK
    * ``data_block_addr<N>`` - int : address of N-th data block
    * ``flags`` - int : data list flags
    * ``reserved1`` - int : reserved bytes
    * ``data_block_nr`` - int : number of data blocks referenced by thsi list

    DLBLOCK specific fields

    * for equall lenght blocks

        * ``data_block_len`` - int : equall uncompressed size in bytes for all
          referenced data blocks; last block can be smaller

    * for variable lenght blocks

        * ``offset_<N>`` - int : byte offset of N-th data block

    Other attributes

    * ``address`` - int : data list address

    """

    def __init__(self, **kwargs):

        try:
            self.address = address = kwargs["address"]
            stream = kwargs["stream"]
            mapped = kwargs.get("mapped", False)

            if mapped:
                (self.id, self.reserved0, self.block_len, self.links_nr) = COMMON_uf(
                    stream, address
                )

                if self.id != b"##DL":
                    message = f'Expected "##DL" block @{hex(address)} but found "{self.id}"'

                    logger.exception(message)
                    raise MdfException(message)

                address += COMMON_SIZE

                links = unpack_from(
                    f"<{self.links_nr}Q", stream, address
                )

                self.next_dl_addr = links[0]

                for i, addr in enumerate(links[1:]):
                    self[f"data_block_addr{i}"] = addr

                stream.seek(address + self.links_nr * 8)

                self.flags = stream.read(1)[0]
                if self.flags & v4c.FLAG_DL_EQUAL_LENGHT:
                    (self.reserved1, self.data_block_nr, self.data_block_len) = unpack(
                        "<3sIQ", stream.read(15)
                    )
                else:
                    (self.reserved1, self.data_block_nr) = unpack("<3sI", stream.read(7))
                    offsets = unpack(
                        "<{}Q".format(self.links_nr - 1),
                        stream.read((self.links_nr - 1) * 8),
                    )
                    for i, offset in enumerate(offsets):
                        self[f"offset_{i}"] = offset
            else:

                stream.seek(address)

                (self.id, self.reserved0, self.block_len, self.links_nr) = COMMON_u(
                    stream.read(COMMON_SIZE)
                )

                if self.id != b"##DL":
                    message = f'Expected "##DL" block @{hex(address)} but found "{self.id}"'

                    logger.exception(message)
                    raise MdfException(message)

                links = unpack(
                    f"<{self.links_nr}Q", stream.read(self.links_nr * 8)
                )

                self.next_dl_addr = links[0]

                for i, addr in enumerate(links[1:]):
                    self[f"data_block_addr{i}"] = addr

                self.flags = stream.read(1)[0]
                if self.flags & v4c.FLAG_DL_EQUAL_LENGHT:
                    (self.reserved1, self.data_block_nr, self.data_block_len) = unpack(
                        "<3sIQ", stream.read(15)
                    )
                else:
                    (self.reserved1, self.data_block_nr) = unpack("<3sI", stream.read(7))
                    offsets = unpack(
                        "<{}Q".format(self.links_nr - 1),
                        stream.read((self.links_nr - 1) * 8),
                    )
                    for i, offset in enumerate(offsets):
                        self[f"offset_{i}"] = offset

        except KeyError:

            self.address = 0
            self.id = b"##DL"
            self.reserved0 = 0
            self.block_len = 40 + 8 * kwargs.get("links_nr", 2)
            self.links_nr = kwargs.get("links_nr", 2)
            self.next_dl_addr = 0

            for i in range(self.links_nr - 1):
                self[f"data_block_addr{i}"] = kwargs.get(f"data_block_addr{i}", 0)

            self.flags = kwargs.get("flags", 1)
            self.reserved1 = kwargs.get("reserved1", b"\0\0\0")
            self.data_block_nr = kwargs.get("data_block_nr", 1)
            if self.flags & v4c.FLAG_DL_EQUAL_LENGHT:
                self.data_block_len = kwargs["data_block_len"]
            else:
                for i, offset in enumerate(self.links_nr - 1):
                    self[f"offset_{i}"] = kwargs[f"offset_{i}"]

    def __getitem__(self, item):
        return self.__getattribute__(item)

    def __setitem__(self, item, value):
        self.__setattr__(item, value)

    def __bytes__(self):
        fmt = v4c.FMT_DATA_LIST.format(self.links_nr)
        keys = ("id", "reserved0", "block_len", "links_nr", "next_dl_addr")
        keys += tuple(f"data_block_addr{i}" for i in range(self.links_nr - 1))
        keys += ("flags", "reserved1", "data_block_nr", "data_block_len")
        result = pack(fmt, *[getattr(self, key) for key in keys])
        return result


class _EventBlockBase:
    __slots__ = (
        "address",
        "comment",
        "name",
        "parent",
        "range_start",
        "scopes",
        "id",
        "reserved0",
        "block_len",
        "links_nr",
        "next_ev_addr",
        "parent_ev_addr",
        "range_start_ev_addr",
        "name_addr",
        "comment_addr",
        "event_type",
        "sync_type",
        "range_type",
        "cause",
        "flags",
        "reserved1",
        "scope_nr",
        "attachment_nr",
        "creator_index",
        "sync_base",
        "sync_factor",
    )


class EventBlock(_EventBlockBase):
    """
    *EventBlock* has the following attributes, that are also available as
    dict like key-value pairs

    EVBLOCK fields

    * ``id`` - bytes : block ID; always b'##EV'
    * ``reserved0`` - int : reserved bytes
    * ``block_len`` - int : block bytes size
    * ``links_nr`` - int : number of links
    * ``next_ev_addr`` - int : address of next EVBLOCK
    * ``parent_ev_addr`` - int : address of parent EVLBOCK
    * ``range_start_ev_addr`` - int : address of EVBLOCK that is the start of
      the range for which this event is the end
    * ``name_addr`` - int : address of TXBLOCK that contains the event name
    * ``comment_addr`` - int : address of TXBLOCK/MDBLOCK that contains the
      event comment
    * ``scope_<N>_addr`` - int : address of N-th block that represents a scope
      for this event (can be CGBLOCK, CHBLOCK, DGBLOCK)
    * ``attachemnt_<N>_addr`` - int : address of N-th attachment referenced by
      this event
    * ``event_type`` - int : integer code for event type
    * ``sync_type`` - int : integer code for event sync type
    * ``range_type`` - int : integer code for event range type
    * ``cause`` - int : integer code for event cause
    * ``flags`` - int : event flags
    * ``reserved1`` - int : reserved bytes
    * ``scope_nr`` - int : number of scopes referenced by this event
    * ``attachment_nr`` - int : number of attachments referenced by this event
    * ``creator_index`` - int : index of FHBLOCK
    * ``sync_base`` - int : timestamp base value
    * ``sync_factor`` - float : timestamp factor

    Other attributes

    * ``address`` - int : event block address
    * ``comment`` - str : event comment
    * ``name`` - str : event name
    * ``parent`` - int : index of event block that is the parent for the
      current event
    * ``range_start`` - int : index of event block that is the start of the
      range for which the current event is the end
    * ``scopes`` - list : list of (group index, channel index) or channel group
      index that define the scope of the current event

    """

    def __init__(self, **kwargs):

        self.name = self.comment = ""
        self.scopes = []
        self.parent = None
        self.range_start = None

        if "stream" in kwargs:

            self.address = address = kwargs["address"]
            stream = kwargs["stream"]
            stream.seek(address)

            (self.id, self.reserved0, self.block_len, self.links_nr) = COMMON_u(
                stream.read(COMMON_SIZE)
            )

            block = stream.read(self.block_len - COMMON_SIZE)

            links_nr = self.links_nr

            links = unpack_from(f"<{links_nr}Q", block)
            params = unpack_from(v4c.FMT_EVENT_PARAMS, block, links_nr * 8)

            (
                self.next_ev_addr,
                self.parent_ev_addr,
                self.range_start_ev_addr,
                self.name_addr,
                self.comment_addr,
            ) = links[:5]

            scope_nr = params[6]
            for i in range(scope_nr):
                self[f"scope_{i}_addr"] = links[5 + i]

            attachment_nr = params[7]
            for i in range(attachment_nr):
                self[f"attachment_{i}_addr"] = links[5 + scope_nr + i]

            (
                self.event_type,
                self.sync_type,
                self.range_type,
                self.cause,
                self.flags,
                self.reserved1,
                self.scope_nr,
                self.attachment_nr,
                self.creator_index,
                self.sync_base,
                self.sync_factor,
            ) = params

            if self.id != b"##EV":
                message = f'Expected "##EV" block @{hex(address)} but found "{self.id}"'

                logger.exception(message)
                raise MdfException(message)

            self.name = get_text_v4(self.name_addr, stream)
            self.comment = get_text_v4(self.comment_addr, stream)

        else:
            self.address = 0

            scopes = 0
            while f"scope_{scopes}_addr" in kwargs:
                scopes += 1

            self.id = b"##EV"
            self.reserved0 = 0
            self.block_len = 56 + (scopes + 5) * 8
            self.links_nr = scopes + 5
            self.next_ev_addr = kwargs.get("next_ev_addr", 0)
            self.parent_ev_addr = kwargs.get("parent_ev_addr", 0)
            self.range_start_ev_addr = kwargs.get("range_start_ev_addr", 0)
            self.name_addr = kwargs.get("name_addr", 0)
            self.comment_addr = kwargs.get("comment_addr", 0)

            for i in range(scopes):
                self[f"scope_{i}_addr"] = kwargs[f"scope_{i}_addr"]

            self.event_type = kwargs.get("event_type", v4c.EVENT_TYPE_TRIGGER)
            self.sync_type = kwargs.get("sync_type", v4c.EVENT_SYNC_TYPE_S)
            self.range_type = kwargs.get("range_type", v4c.EVENT_RANGE_TYPE_POINT)
            self.cause = kwargs.get("cause", v4c.EVENT_CAUSE_TOOL)
            self.flags = kwargs.get("flags", v4c.FLAG_EV_POST_PROCESSING)
            self.reserved1 = b"\x00\x00\x00"
            self.scope_nr = scopes
            self.attachment_nr = 0
            self.creator_index = 0
            self.sync_base = kwargs.get("sync_base", 0)
            self.sync_factor = kwargs.get("sync_factor", 1.0)

    def update_references(self, ch_map, cg_map):
        self.scopes.clear()
        for i in range(self.scope_nr):
            addr = self[f"scope_{i}_addr"]
            if addr in ch_map:
                self.scopes.append(ch_map[addr])
            elif addr in cg_map:
                self.scopes.append(cg_map[addr])
            else:
                message = (
                    "{} is not a valid CNBLOCK or CGBLOCK "
                    "address for the event scope"
                )
                message = message.format(hex(addr))
                logger.exception(message)
                raise MdfException(message)

    def __bytes__(self):

        fmt = v4c.FMT_EVENT.format(self.links_nr)

        keys = (
            "id",
            "reserved0",
            "block_len",
            "links_nr",
            "next_ev_addr",
            "parent_ev_addr",
            "range_start_ev_addr",
            "name_addr",
            "comment_addr",
        )

        keys += tuple(f"scope_{i}_addr" for i in range(self.scope_nr))

        keys += tuple(f"attachment_{i}_addr" for i in range(self.attachment_nr))

        keys += (
            "event_type",
            "sync_type",
            "range_type",
            "cause",
            "flags",
            "reserved1",
            "scope_nr",
            "attachment_nr",
            "creator_index",
            "sync_base",
            "sync_factor",
        )
        result = pack(fmt, *[getattr(self, key) for key in keys])

        return result

    def __getitem__(self, item):
        return self.__getattribute__(item)

    def __setitem__(self, item, value):
        self.__setattr__(item, value)

    def __str__(self):
        return "EventBlock (name: {}, comment: {}, address: {}, scopes: {}, fields: {})".format(
            self.name, self.comment, hex(self.address), self.scopes, super().__str__()
        )


class FileIdentificationBlock:
    """
    *FileIdentificationBlock* has the following attributes, that are also available as
    dict like key-value pairs

    IDBLOCK fields

    * ``file_identification`` -  bytes : file identifier
    * ``version_str`` - bytes : format identifier
    * ``program_identification`` - bytes : creator program identifier
    * ``reserved0`` - bytes : reserved bytes
    * ``mdf_version`` - int : version number of MDF format
    * ``reserved1`` - bytes : reserved bytes
    * ``unfinalized_standard_flags`` - int : standard flags for unfinalized MDF
    * ``unfinalized_custom_flags`` - int : custom flags for unfinalized MDF

    Other attributes

    * ``address`` - int : should always be 0

    """

    __slots__ = (
        "address",
        "file_identification",
        "version_str",
        "program_identification",
        "reserved0",
        "mdf_version",
        "reserved1",
        "unfinalized_standard_flags",
        "unfinalized_custom_flags",
    )

    def __init__(self, **kwargs):

        super().__init__()

        self.address = 0

        try:

            stream = kwargs["stream"]
            stream.seek(self.address)

            (
                self.file_identification,
                self.version_str,
                self.program_identification,
                self.reserved0,
                self.mdf_version,
                self.reserved1,
                self.unfinalized_standard_flags,
                self.unfinalized_custom_flags,
            ) = unpack(
                v4c.FMT_IDENTIFICATION_BLOCK, stream.read(v4c.IDENTIFICATION_BLOCK_SIZE)
            )

        except KeyError:

            version = kwargs.get("version", "4.00")
            self.file_identification = "MDF     ".encode("utf-8")
            self.version_str = "{}    ".format(version).encode("utf-8")
            self.program_identification = "amdf{}".format(
                __version__.replace(".", "")
            ).encode("utf-8")
            self.reserved0 = b"\0" * 4
            self.mdf_version = int(version.replace(".", ""))
            self.reserved1 = b"\0" * 30
            self.unfinalized_standard_flags = 0
            self.unfinalized_custom_flags = 0

    def __getitem__(self, item):
        return self.__getattribute__(item)

    def __setitem__(self, item, value):
        self.__setattr__(item, value)

    def __bytes__(self):
        result = pack(
            v4c.FMT_IDENTIFICATION_BLOCK,
            *[self[key] for key in v4c.KEYS_IDENTIFICATION_BLOCK],
        )
        return result


class FileHistory:
    """
    *FileHistory* has the following attributes, that are also available as
    dict like key-value pairs

    FHBLOCK fields

    * ``id`` - bytes : block ID; always b'##FH'
    * ``reserved0`` - int : reserved bytes
    * ``block_len`` - int : block bytes size
    * ``links_nr`` - int : number of links
    * ``next_fh_addr`` - int : address of next FHBLOCK
    * ``comment_addr`` - int : address of TXBLOCK/MDBLOCK that contains the
      file history comment
    * ``abs_time`` - int : time stamp at which the file modification happened
    * ``tz_offset`` - int : UTC time offset in hours (= GMT time zone)
    * ``daylight_save_time`` - int : daylight saving time
    * ``time_flags`` - int : time flags
    * ``reserved1`` - bytes : reserved bytes

    Other attributes

    * ``address`` - int : file history address
    * ``comment`` - str : history comment

    """

    __slots__ = (
        "address",
        "comment",
        "id",
        "reserved0",
        "block_len",
        "links_nr",
        "next_fh_addr",
        "comment_addr",
        "abs_time",
        "tz_offset",
        "daylight_save_time",
        "time_flags",
        "reserved1",
    )

    def __init__(self, **kwargs):
        super().__init__()

        self.comment = ""

        try:
            self.address = address = kwargs["address"]
            stream = kwargs["stream"]
            stream.seek(address)

            (
                self.id,
                self.reserved0,
                self.block_len,
                self.links_nr,
                self.next_fh_addr,
                self.comment_addr,
                self.abs_time,
                self.tz_offset,
                self.daylight_save_time,
                self.time_flags,
                self.reserved1,
            ) = unpack(v4c.FMT_FILE_HISTORY, stream.read(v4c.FH_BLOCK_SIZE))

            if self.id != b"##FH":
                message = f'Expected "##FH" block @{hex(address)} but found "{self.id}"'

                logger.exception(message)
                raise MdfException(message)

            self.comment = get_text_v4(address=self.comment_addr, stream=stream)

        except KeyError:
            self.id = b"##FH"
            self.reserved0 = kwargs.get("reserved0", 0)
            self.block_len = kwargs.get("block_len", v4c.FH_BLOCK_SIZE)
            self.links_nr = kwargs.get("links_nr", 2)
            self.next_fh_addr = kwargs.get("next_fh_addr", 0)
            self.comment_addr = kwargs.get("comment_addr", 0)
            self.abs_time = kwargs.get("abs_time", int(time.time()) * 10 ** 9)
            self.tz_offset = kwargs.get("tz_offset", 120)
            self.daylight_save_time = kwargs.get("daylight_save_time", 60)
            self.time_flags = kwargs.get("time_flags", 2)
            self.reserved1 = kwargs.get("reserved1", b"\x00" * 3)

    def to_blocks(self, address, blocks, defined_texts):
        text = self.comment
        if text:
            if text in defined_texts:
                self.comment_addr = defined_texts[text]
            else:
                meta = text.startswith("<FHcomment")
                tx_block = TextBlock(text=text, meta=meta)
                self.comment_addr = address
                defined_texts[text] = address
                tx_block.address = address
                address += tx_block.block_len
                blocks.append(tx_block)
        else:
            self.comment_addr = 0

        blocks.append(self)
        self.address = address
        address += self.block_len

        return address

    def __getitem__(self, item):
        return self.__getattribute__(item)

    def __setitem__(self, item, value):
        self.__setattr__(item, value)

    def __bytes__(self):
        result = pack(
            v4c.FMT_FILE_HISTORY, *[self[key] for key in v4c.KEYS_FILE_HISTORY]
        )
        return result


class HeaderBlock:
    """
    *HeaderBlock* has the following attributes, that are also available as
    dict like key-value pairs

    HDBLOCK fields

    * ``id`` - bytes : block ID; always b'##HD'
    * ``reserved0`` - int : reserved bytes
    * ``block_len`` - int : block bytes size
    * ``links_nr`` - int : number of links
    * ``first_dg_addr`` - int : address of first DGBLOCK
    * ``file_history_addr`` - int : address of first FHBLOCK
    * ``channel_tree_addr`` - int : address of first CHBLOCK
    * ``first_attachment_addr`` - int : address of first ATBLOCK
    * ``first_event_addr`` - int : address of first EVBLOCK
    * ``comment_addr`` - int : address of TXBLOCK/MDBLOCK that contains the
      file comment
    * ``abs_time`` - int : time stamp at which recording was started in
      nanoseconds.
    * ``tz_offset`` - int : UTC time offset in hours (= GMT time zone)
    * ``daylight_save_time`` - int : daylight saving time
    * ``time_flags`` - int : time flags
    * ``time_quality`` - int : time quality flags
    * ``flags`` - int : file flags
    * ``reserved1`` - int : reserved bytes
    * ``start_angle`` - int : angle value at measurement start
    * ``start_distance`` - int : distance value at measurement start

    Other attributes

    * ``address`` - int : header address
    * ``comment`` - str : file comment
    * ``author`` - str : measurement author
    * ``department`` - str : author's department
    * ``project`` - str : working project
    * ``subject`` - str : measurement subject

    """

    __slots__ = (
        "address",
        "comment",
        "author",
        "department",
        "project",
        "subject",
        "id",
        "reserved0",
        "block_len",
        "links_nr",
        "first_dg_addr",
        "file_history_addr",
        "channel_tree_addr",
        "first_attachment_addr",
        "first_event_addr",
        "comment_addr",
        "abs_time",
        "tz_offset",
        "daylight_save_time",
        "time_flags",
        "time_quality",
        "flags",
        "reserved1",
        "start_angle",
        "start_distance",
    )

    def __init__(self, **kwargs):
        super().__init__()

        self.comment = ""

        self.author = self.project = self.subject = self.department = ""

        try:
            self.address = address = kwargs["address"]
            stream = kwargs["stream"]
            stream.seek(address)

            (
                self.id,
                self.reserved0,
                self.block_len,
                self.links_nr,
                self.first_dg_addr,
                self.file_history_addr,
                self.channel_tree_addr,
                self.first_attachment_addr,
                self.first_event_addr,
                self.comment_addr,
                self.abs_time,
                self.tz_offset,
                self.daylight_save_time,
                self.time_flags,
                self.time_quality,
                self.flags,
                self.reserved1,
                self.start_angle,
                self.start_distance,
            ) = unpack(v4c.FMT_HEADER_BLOCK, stream.read(v4c.HEADER_BLOCK_SIZE))

            if self.id != b"##HD":
                message = f'Expected "##HD" block @{hex(address)} but found "{self.id}"'

                logger.exception(message)
                raise MdfException(message)

            self.comment = get_text_v4(address=self.comment_addr, stream=stream)

        except KeyError:

            self.address = 0x40
            self.id = b"##HD"
            self.reserved0 = kwargs.get("reserved3", 0)
            self.block_len = kwargs.get("block_len", v4c.HEADER_BLOCK_SIZE)
            self.links_nr = kwargs.get("links_nr", 6)
            self.first_dg_addr = kwargs.get("first_dg_addr", 0)
            self.file_history_addr = kwargs.get("file_history_addr", 0)
            self.channel_tree_addr = kwargs.get("channel_tree_addr", 0)
            self.first_attachment_addr = kwargs.get("first_attachment_addr", 0)
            self.first_event_addr = kwargs.get("first_event_addr", 0)
            self.comment_addr = kwargs.get("comment_addr", 0)
            self.abs_time = kwargs.get("abs_time", int(time.time()) * 10 ** 9)
            self.tz_offset = kwargs.get("tz_offset", 120)
            self.daylight_save_time = kwargs.get("daylight_save_time", 60)
            self.time_flags = kwargs.get("time_flags", 2)
            self.time_quality = kwargs.get("time_quality", 0)
            self.flags = kwargs.get("flags", 0)
            self.reserved1 = kwargs.get("reserved4", 0)
            self.start_angle = kwargs.get("start_angle", 0)
            self.start_distance = kwargs.get("start_distance", 0)

        if self.comment.startswith("<HDcomment"):
            comment = self.comment
            comment_xml = ET.fromstring(comment)
            common_properties = comment_xml.find(".//common_properties")
            if common_properties is not None:
                for e in common_properties:
                    name = e.get("name")
                    if name == "author":
                        self.author = e.text
                    elif name == "department":
                        self.department = e.text
                    elif name == "project":
                        self.project = e.text
                    elif name == "subject":
                        self.subject = e.text

    def __getitem__(self, item):
        return self.__getattribute__(item)

    def __setitem__(self, item, value):
        self.__setattr__(item, value)

    @property
    def start_time(self):
        """ getter and setter the measurement start timestamp

        Returns
        -------
        timestamp : datetime.datetime
            start timestamp

        """

        timestamp = self.abs_time / 10 ** 9
        try:
            timestamp = datetime.fromtimestamp(timestamp)
        except OSError:
            timestamp = datetime.now()

        return timestamp

    @start_time.setter
    def start_time(self, timestamp):
        timestamp = timestamp - datetime(1970, 1, 1)
        timestamp = int(timestamp.total_seconds() * 10 ** 9)
        self.abs_time = timestamp
        self.tz_offset = 0
        self.daylight_save_time = 0

    def to_blocks(self, address, blocks):
        blocks.append(self)
        self.address = address
        address += self.block_len

        if self.comment.startswith("<HDcomment"):
            comment = self.comment
            comment = ET.fromstring(comment)
            common_properties = comment.find(".//common_properties")
            if common_properties is not None:
                for e in common_properties:
                    name = e.get("name")
                    if name == "author":
                        e.text = self.author
                        break
                else:
                    author = ET.SubElement(
                        common_properties, "e", name="author"
                    ).text = self.author

                for e in common_properties:
                    name = e.get("name")
                    if name == "department":
                        e.text = self.department
                        break
                else:
                    department = ET.SubElement(
                        common_properties, "e", name="department"
                    ).text = self.department

                for e in common_properties:
                    name = e.get("name")
                    if name == "project":
                        e.text = self.author
                        break
                else:
                    project = ET.SubElement(
                        common_properties, "e", name="project"
                    ).text = self.project

                for e in common_properties:
                    name = e.get("name")
                    if name == "subject":
                        e.text = self.author
                        break
                else:
                    subject = ET.SubElement(
                        common_properties, "e", name="subject"
                    ).text = self.subject

            else:
                common_properties = ET.SubElement(comment, "common_properties")
                author = ET.SubElement(
                    common_properties, "e", name="author"
                ).text = self.author
                department = ET.SubElement(
                    common_properties, "e", name="department"
                ).text = self.department
                project = ET.SubElement(
                    common_properties, "e", name="project"
                ).text = self.project
                subject = ET.SubElement(
                    common_properties, "e", name="subject"
                ).text = self.subject

            comment = ET.tostring(comment, encoding="utf8", method="xml").replace(b"<?xml version='1.0' encoding='utf8'?>\n", b"")

        else:
            comment = v4c.HD_COMMENT_TEMPLATE.format(
                self.comment, self.author, self.department, self.project, self.subject
            )

        tx_block = TextBlock(text=comment, meta=True)
        self.comment_addr = address
        tx_block.address = address
        address += tx_block.block_len
        blocks.append(tx_block)

        return address

    def __bytes__(self):
        result = pack(
            v4c.FMT_HEADER_BLOCK, *[self[key] for key in v4c.KEYS_HEADER_BLOCK]
        )
        return result


class HeaderList:
    """
    *HeaderList* has the following attributes, that are also available as
    dict like key-value pairs

    HLBLOCK fields

    * ``id`` - bytes : block ID; always b'##HL'
    * ``reserved0`` - int : reserved bytes
    * ``block_len`` - int : block bytes size
    * ``links_nr`` - int : number of links
    * ``first_dl_addr`` - int : address of first data list block for this header
      list
    * ``flags`` - int : source flags
    * ``zip_type`` - int : integer code for zip type
    * ``reserved1`` - bytes : reserved bytes

    Other attributes

    * ``address`` - int : header list address

    """

    __slots__ = (
        "address",
        "id",
        "reserved0",
        "block_len",
        "links_nr",
        "first_dl_addr",
        "flags",
        "zip_type",
        "reserved1",
    )

    def __init__(self, **kwargs):
        super().__init__()

        try:
            self.address = address = kwargs["address"]
            stream = kwargs["stream"]
            stream.seek(address)

            (
                self.id,
                self.reserved0,
                self.block_len,
                self.links_nr,
                self.first_dl_addr,
                self.flags,
                self.zip_type,
                self.reserved1,
            ) = unpack(v4c.FMT_HL_BLOCK, stream.read(v4c.HL_BLOCK_SIZE))

            if self.id != b"##HL":
                message = f'Expected "##HL" block @{hex(address)} but found "{self.id}"'

                logger.exception(message)
                raise MdfException(message)

        except KeyError:

            self.address = 0
            self.id = b"##HL"
            self.reserved0 = 0
            self.block_len = v4c.HL_BLOCK_SIZE
            self.links_nr = 1
            self.first_dl_addr = kwargs.get("first_dl_addr", 0)
            self.flags = 1
            self.zip_type = kwargs.get("zip_type", 0)
            self.reserved1 = b"\x00" * 5

    def __getitem__(self, item):
        return self.__getattribute__(item)

    def __setitem__(self, item, value):
        self.__setattr__(item, value)

    def __bytes__(self):
        result = pack(v4c.FMT_HL_BLOCK, *[self[key] for key in v4c.KEYS_HL_BLOCK])
        return result


class SourceInformation:
    """
    *SourceInformation* has the following attributes, that are also available as
    dict like key-value pairs

    SIBLOCK fields

    * ``id`` - bytes : block ID; always b'##SI'
    * ``reserved0`` - int : reserved bytes
    * ``block_len`` - int : block bytes size
    * ``links_nr`` - int : number of links
    * ``name_addr`` - int : address of TXBLOCK that contains the source name
    * ``path_addr`` - int : address of TXBLOCK that contains the source path
    * ``comment_addr`` - int : address of TXBLOCK/MDBLOCK tha contains the
      source comment
    * ``source_type`` - int : integer code for source type
    * ``bus_type`` - int : integer code for source bus type
    * ``flags`` - int : source flags
    * ``reserved1`` - bytes : reserved bytes

    Other attributes

    * ``address`` - int : source information address
    * ``comment`` - str : source comment
    * ``name`` - str : source name
    * ``path`` - str : source path

    """

    __slots__ = (
        "address",
        "comment",
        "name",
        "path",
        "id",
        "reserved0",
        "block_len",
        "links_nr",
        "name_addr",
        "path_addr",
        "comment_addr",
        "source_type",
        "bus_type",
        "flags",
        "reserved1",
    )

    def __init__(self, **kwargs):
        self.name = self.path = self.comment = ""

        if "stream" in kwargs:

            stream = kwargs["stream"]
            mapped = kwargs.get("mapped", False)
            try:
                block = kwargs["raw_bytes"]
                self.address = kwargs.get("address", 0)
            except KeyError:
                self.address = address = kwargs["address"]
                stream.seek(address)
                block = stream.read(v4c.SI_BLOCK_SIZE)

            (
                self.id,
                self.reserved0,
                self.block_len,
                self.links_nr,
                self.name_addr,
                self.path_addr,
                self.comment_addr,
                self.source_type,
                self.bus_type,
                self.flags,
                self.reserved1,
            ) = unpack(v4c.FMT_SOURCE_INFORMATION, block)

            if self.id != b"##SI":
                message = f'Expected "##SI" block @{hex(address)} but found "{self.id}"'

                logger.exception(message)
                raise MdfException(message)

            self.name = get_text_v4(address=self.name_addr, stream=stream, mapped=mapped)
            self.path = get_text_v4(address=self.path_addr, stream=stream, mapped=mapped)
            self.comment = get_text_v4(address=self.comment_addr, stream=stream, mapped=mapped)

        else:
            self.address = 0
            self.id = b"##SI"
            self.reserved0 = 0
            self.block_len = v4c.SI_BLOCK_SIZE
            self.links_nr = 3
            self.name_addr = 0
            self.path_addr = 0
            self.comment_addr = 0
            self.source_type = kwargs.get("source_type", v4c.SOURCE_TOOL)
            self.bus_type = kwargs.get("bus_type", v4c.BUS_TYPE_NONE)
            self.flags = 0
            self.reserved1 = b"\x00" * 5

    def __getitem__(self, item):
        return self.__getattribute__(item)

    def __setitem__(self, item, value):
        self.__setattr__(item, value)

    def __contains__(self, item):
        return hasattr(self, item)

    def metadata(self):
        max_len = max(len(key) for key in v4c.KEYS_SOURCE_INFORMATION)
        template = f"{{: <{max_len}}}: {{}}"

        metadata = []
        lines = f"""
name: {self.name}
path: {self.path}
address: {hex(self.address)}
comment: {self.comment}

""".split("\n")
        for key in v4c.KEYS_SOURCE_INFORMATION:
            val = getattr(self, key)
            if key.endswith("addr") or key.startswith("text_"):
                lines.append(template.format(key, hex(val)))
            elif isinstance(val, float):
                lines.append(template.format(key, round(val, 6)))
            else:
                if isinstance(val, bytes):
                    lines.append(template.format(key, val.strip(b"\0")))
                else:
                    lines.append(template.format(key, val))
        for line in lines:
            if not line:
                metadata.append(line)
            else:
                for wrapped_line in wrap(line, width=120):
                    metadata.append(wrapped_line)

        return "\n".join(metadata)

    def to_blocks(self, address, blocks, defined_texts, si_map):
        text = self.name
        if text:
            if text in defined_texts:
                self.name_addr = defined_texts[text]
            else:
                tx_block = TextBlock(text=text)
                self.name_addr = address
                defined_texts[text] = address
                tx_block.address = address
                address += tx_block.block_len
                blocks.append(tx_block)
        else:
            self.name_addr = 0

        text = self.path
        if text:
            if text in defined_texts:
                self.path_addr = defined_texts[text]
            else:
                tx_block = TextBlock(text=text)
                self.path_addr = address
                defined_texts[text] = address
                tx_block.address = address
                address += tx_block.block_len
                blocks.append(tx_block)
        else:
            self.path_addr = 0

        text = self.comment
        if text:
            if text in defined_texts:
                self.comment_addr = defined_texts[text]
            else:
                meta = text.startswith("<SIcomment")
                tx_block = TextBlock(text=text, meta=meta)
                self.comment_addr = address
                defined_texts[text] = address
                tx_block.address = address
                address += tx_block.block_len
                blocks.append(tx_block)
        else:
            self.comment_addr = 0

        bts = bytes(self)
        if bts in si_map:
            self.address = si_map[bts]
        else:
            blocks.append(bts)
            si_map[bts] = address
            self.address = address
            address += self.block_len

        return address

    def to_common_source(self):
        return SignalSource(
            self.name, self.path, self.comment, self.source_type, self.bus_type
        )

    def __bytes__(self):
        return v4c.SOURCE_INFORMATION_PACK(
            self.id,
            self.reserved0,
            self.block_len,
            self.links_nr,
            self.name_addr,
            self.path_addr,
            self.comment_addr,
            self.source_type,
            self.bus_type,
            self.flags,
            self.reserved1,
        )

    def __str__(self):
        return "<SourceInformation (name: {}, path: {}, comment: {}, address: {}, fields: {})>".format(
            self.name, self.path, self.comment, hex(self.address), block_fields(self)
        )


class TextBlock:
    """common TXBLOCK and MDBLOCK class

    *TextBlock* has the following attributes, that are also available as
    dict like key-value pairs

    TXBLOCK fields

    * ``id`` - bytes : block ID; b'##TX' for TXBLOCK and b'##MD' for MDBLOCK
    * ``reserved0`` - int : reserved bytes
    * ``block_len`` - int : block bytes size
    * ``links_nr`` - int : number of links
    * ``text`` - bytes : actual text content

    Other attributes

    * ``address`` - int : text block address

    Parameters
    ----------
    address : int
        block address
    stream : handle
        file handle
    meta : bool
        flag to set the block type to MDBLOCK for dynamically created objects; default *False*
    text : bytes/str
        text content for dynamically created objects


    """

    __slots__ = ("address", "id", "reserved0", "block_len", "links_nr", "text")

    def __init__(self, **kwargs):
        super().__init__()

        if "stream" in kwargs:
            stream = kwargs["stream"]
            mapped = kwargs.get("mapped", False)
            self.address = address = kwargs["address"]

            if mapped:
                (self.id, self.reserved0, self.block_len, self.links_nr) = COMMON_uf(
                    stream, address
                )

                size = self.block_len - COMMON_SIZE

                if self.id not in (b"##TX", b"##MD"):
                    message = f'Expected "##TX" or "##MD" block @{hex(address)} but found "{self.id}"'
                    logger.exception(message)
                    raise MdfException(message)

                self.text = text = stream[address + COMMON_SIZE: address + self.block_len]

            else:
                stream.seek(address)
                (self.id, self.reserved0, self.block_len, self.links_nr) = COMMON_u(
                    stream.read(COMMON_SIZE)
                )

                size = self.block_len - COMMON_SIZE

                if self.id not in (b"##TX", b"##MD"):
                    message = f'Expected "##TX" or "##MD" block @{hex(address)} but found "{self.id}"'
                    logger.exception(message)
                    raise MdfException(message)

                self.text = text = stream.read(size)

            align = size % 8
            if align:
                self.block_len = size + COMMON_SIZE + 8 - align
            else:
                if text:
                    if text[-1]:
                        self.block_len += 8
                else:
                    self.block_len += 8

        else:

            self.address = 0
            text = kwargs["text"]

            try:
                text = text.encode("utf-8", "replace")
            except AttributeError:
                pass

            size = len(text)

            self.id = b"##MD" if kwargs.get("meta", False) else b"##TX"
            self.reserved0 = 0
            self.links_nr = 0
            self.text = text

            self.block_len = size + 32 - size % 8

    def __getitem__(self, item):
        return self.__getattribute__(item)

    def __setitem__(self, item, value):
        self.__setattr__(item, value)

    def __bytes__(self):
        return v4c.COMMON_p(
            self.id, self.reserved0, self.block_len, self.links_nr
        ) + pack(f"{self.block_len - COMMON_SIZE}s", self.text)

    def __repr__(self):
        return (
            f"TextBlock(id={self.id},"
            f"reserved0={self.reserved0}, "
            f"block_len={self.block_len}, "
            f"links_nr={self.links_nr} "
            f"text={self.text})"
        )

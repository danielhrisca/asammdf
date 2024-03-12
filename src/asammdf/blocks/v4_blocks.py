"""
classes that implement the blocks for MDF version 4
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import md5
import logging
from pathlib import Path
import re
from struct import pack, unpack, unpack_from
from textwrap import wrap
import time
from traceback import format_exc
from typing import Any, TYPE_CHECKING
from xml.dom import minidom
import xml.etree.ElementTree as ET

import dateutil.tz

try:
    from isal.isal_zlib import compress, decompress

    COMPRESSION_LEVEL = 2

except ImportError:
    from zlib import compress, decompress

    COMPRESSION_LEVEL = 1


from numexpr import evaluate

try:
    from sympy import lambdify, symbols

except:
    lambdify, symbols = None, None

import numpy as np

from ..version import __version__
from . import v4_constants as v4c
from .utils import (
    block_fields,
    escape_xml_string,
    extract_display_names,
    extract_ev_tool,
    FLOAT64_u,
    get_text_v4,
    is_file_like,
    MdfException,
    UINT8_uf,
    UINT64_u,
    UINT64_uf,
)

if TYPE_CHECKING:
    from .source_utils import Source

SEEK_START = v4c.SEEK_START
SEEK_END = v4c.SEEK_END
COMMON_SIZE = v4c.COMMON_SIZE
COMMON_u = v4c.COMMON_u
COMMON_uf = v4c.COMMON_uf

CN_BLOCK_SIZE = v4c.CN_BLOCK_SIZE
CN_SINGLE_ATTACHMENT_BLOCK_SIZE = v4c.CN_SINGLE_ATTACHMENT_BLOCK_SIZE
SIMPLE_CHANNEL_PARAMS_uf = v4c.SIMPLE_CHANNEL_PARAMS_uf
SINGLE_ATTACHMENT_CHANNEL_PARAMS_uf = v4c.SINGLE_ATTACHMENT_CHANNEL_PARAMS_uf

EIGHT_BYTES = bytes(8)

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
    * ``embedded_data`` - bytes : embedded attachment bytes

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

    def __init__(self, **kwargs) -> None:
        self.file_name = self.mime = self.comment = ""

        try:
            self.address = address = kwargs["address"]
            stream = kwargs["stream"]
            mapped = kwargs.get("mapped", False) or not is_file_like(stream)

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

                self.embedded_data = stream[address : address + self.embedded_size]
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
            self.address = 0

            self.comment = kwargs.get("comment", "")
            self.mime = kwargs.get("mime", "")

            file_name = Path(kwargs.get("file_name", None) or "bin.bin")

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
                    data = compress(data, COMPRESSION_LEVEL)
                    embedded_size = len(data)
                self.file_name = file_name.name

            else:
                self.file_name = str(file_name)
                if len(data) > 0:
                    file_name.write_bytes(data)
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
            self.creator_index = kwargs.get("creator_index", 0)
            self.reserved1 = 0
            self.md5_sum = md5_sum
            self.original_size = original_size
            self.embedded_size = embedded_size
            self.embedded_data = data

    def extract(self) -> bytes:
        """extract attachment data

        Returns
        -------
        data : bytes
        """
        if self.flags & v4c.FLAG_AT_EMBEDDED:
            if self.flags & v4c.FLAG_AT_COMPRESSED_EMBEDDED:
                data = decompress(self.embedded_data, bufsize=self.original_size)
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

    def to_blocks(self, address: int, blocks: list[Any], defined_texts: dict[str, int]) -> int:
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
        if align:
            blocks.append(b"\0" * (8 - align))
            address += 8 - align

        return address

    def __getitem__(self, item: str) -> Any:
        return self.__getattribute__(item)

    def __setitem__(self, item: str, value: Any) -> None:
        self.__setattr__(item, value)

    def __bytes__(self) -> bytes:
        fmt = f"{v4c.FMT_AT_COMMON}{self.embedded_size}s"
        result = pack(fmt, *[self[key] for key in v4c.KEYS_AT_BLOCK])
        return result

    def __repr__(self):
        return f"ATBLOCK(address={self.address:x}, file_name={self.file_name}, comment={self.comment})"


CN = b"##CN"


class Channel:
    """If the `load_metadata` keyword argument is not provided or is False,
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
    * ``precision`` - int : integer code for the precision
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
      the index reference to the attachment block index
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
        "display_names",
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
        "standard_C_size",
    )

    def __init__(self, **kwargs) -> None:
        if "stream" in kwargs:
            self.address = address = kwargs["address"]
            self.dtype_fmt = self.attachment = None
            stream = kwargs["stream"]
            mapped = kwargs["mapped"]

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

                elif self.block_len == CN_SINGLE_ATTACHMENT_BLOCK_SIZE:
                    (
                        self.next_ch_addr,
                        self.component_addr,
                        self.name_addr,
                        self.source_addr,
                        self.conversion_addr,
                        self.data_block_addr,
                        self.unit_addr,
                        self.comment_addr,
                        self.attachment_addr,
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
                    ) = SINGLE_ATTACHMENT_CHANNEL_PARAMS_uf(stream, address + COMMON_SIZE)

                    self.attachment = kwargs["at_map"].get(self.attachment_addr, 0)

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
                        params = list(params)
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

                tx_map = kwargs["tx_map"]

                parsed_strings = kwargs["parsed_strings"]
                if parsed_strings is None:
                    self.name = get_text_v4(self.name_addr, stream, mapped=mapped)
                    self.comment = get_text_v4(self.comment_addr, stream, mapped=mapped)

                    if kwargs["use_display_names"]:
                        self.display_names = extract_display_names(self.comment)
                    else:
                        self.display_names = {}
                else:
                    self.name, self.display_names, self.comment = parsed_strings

                addr = self.unit_addr
                if addr in tx_map:
                    self.unit = tx_map[addr]
                else:
                    self.unit = get_text_v4(addr, stream, mapped=mapped)
                    tx_map[addr] = self.unit

                address = self.conversion_addr
                if address:
                    cc_map = kwargs["cc_map"]
                    try:
                        if address in cc_map:
                            conv = cc_map[address]
                        else:
                            (size,) = UINT64_uf(stream, address + 8)
                            raw_bytes = stream[address : address + size]

                            if raw_bytes in cc_map:
                                conv = cc_map[raw_bytes]
                            else:
                                conv = ChannelConversion(
                                    raw_bytes=raw_bytes,
                                    stream=stream,
                                    address=address,
                                    mapped=mapped,
                                    tx_map=tx_map,
                                )
                                cc_map[raw_bytes] = cc_map[address] = conv
                    except:
                        logger.warning(
                            f"Channel conversion parsing error: {format_exc()}. The error is ignored and the channel conversion is None"
                        )
                        conv = None

                    self.conversion = conv
                else:
                    self.conversion = None

                address = self.source_addr
                if address:
                    si_map = kwargs["si_map"]
                    try:
                        if address in si_map:
                            source = si_map[address]
                        else:
                            raw_bytes = stream[address : address + v4c.SI_BLOCK_SIZE]

                            if raw_bytes in si_map:
                                source = si_map[raw_bytes]
                            else:
                                source = SourceInformation(
                                    raw_bytes=raw_bytes,
                                    stream=stream,
                                    address=address,
                                    mapped=mapped,
                                    tx_map=tx_map,
                                )
                                si_map[raw_bytes] = si_map[address] = source
                    except:
                        logger.warning(
                            f"Channel source parsing error: {format_exc()}. The error is ignored and the channel source is None"
                        )
                        source = None

                    self.source = source
                else:
                    self.source = None

            else:
                stream.seek(address)

                block = stream.read(CN_SINGLE_ATTACHMENT_BLOCK_SIZE)

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

                elif self.block_len == CN_SINGLE_ATTACHMENT_BLOCK_SIZE:
                    (
                        self.next_ch_addr,
                        self.component_addr,
                        self.name_addr,
                        self.source_addr,
                        self.conversion_addr,
                        self.data_block_addr,
                        self.unit_addr,
                        self.comment_addr,
                        self.attachment_addr,
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
                    ) = SINGLE_ATTACHMENT_CHANNEL_PARAMS_uf(block, COMMON_SIZE)
                    at_map = kwargs.get("at_map", {})
                    self.attachment = at_map.get(self.attachment_addr, 0)

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
                        params = list(params)
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

                tx_map = kwargs["tx_map"]
                parsed_strings = kwargs["parsed_strings"]

                if parsed_strings is None:
                    self.name = get_text_v4(self.name_addr, stream)
                    self.comment = get_text_v4(self.comment_addr, stream)

                    if kwargs["use_display_names"]:
                        self.display_names = extract_display_names(self.comment)
                    else:
                        self.display_names = {}
                else:
                    self.name, self.display_names, self.comment = parsed_strings
                addr = self.unit_addr
                if addr in tx_map:
                    self.unit = tx_map[addr]
                else:
                    self.unit = get_text_v4(addr, stream)
                    tx_map[addr] = self.unit

                si_map = kwargs["si_map"]
                cc_map = kwargs["cc_map"]

                address = self.conversion_addr
                if address:
                    try:
                        if address in cc_map:
                            conv = cc_map[address]
                        else:
                            stream.seek(address + 8)
                            (size,) = UINT64_u(stream.read(8))
                            stream.seek(address)
                            raw_bytes = stream.read(size)
                            if raw_bytes in cc_map:
                                conv = cc_map[raw_bytes]
                            else:
                                conv = ChannelConversion(
                                    raw_bytes=raw_bytes,
                                    stream=stream,
                                    address=address,
                                    tx_map=tx_map,
                                    mapped=mapped,
                                )
                                cc_map[raw_bytes] = cc_map[address] = conv
                    except:
                        logger.warning(
                            f"Channel conversion parsing error: {format_exc()}. The error is ignored and the channel conversion is None"
                        )
                        conv = None

                    self.conversion = conv
                else:
                    self.conversion = None

                address = self.source_addr
                if address:
                    try:
                        if address in si_map:
                            source = si_map[address]
                        else:
                            stream.seek(address)
                            raw_bytes = stream.read(v4c.SI_BLOCK_SIZE)
                            if raw_bytes in si_map:
                                source = si_map[raw_bytes]
                            else:
                                source = SourceInformation(
                                    raw_bytes=raw_bytes,
                                    stream=stream,
                                    address=address,
                                    tx_map=tx_map,
                                    mapped=mapped,
                                )
                                si_map[raw_bytes] = si_map[address] = source
                    except:
                        logger.warning(
                            f"Channel source parsing error: {format_exc()}. The error is ignored and the channel source is None"
                        )
                        source = None

                    self.source = source
                else:
                    self.source = None
        else:
            self.address = 0
            self.name = self.comment = self.unit = ""
            self.display_names = {}
            self.conversion = self.source = self.attachment = self.dtype_fmt = None

            (
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
            ) = (
                b"##CN",
                0,
                v4c.CN_BLOCK_SIZE,
                8,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                kwargs["channel_type"],
                kwargs.get("sync_type", 0),
                kwargs["data_type"],
                kwargs["bit_offset"],
                kwargs["byte_offset"],
                kwargs["bit_count"],
                kwargs.get("flags", 0),
                kwargs.get("pos_invalidation_bit", 0),
                kwargs.get("precision", 3),
                0,
                0,
                kwargs.get("min_raw_value", 0),
                kwargs.get("max_raw_value", 0),
                kwargs.get("lower_limit", 0),
                kwargs.get("upper_limit", 0),
                kwargs.get("lower_ext_limit", 0),
                kwargs.get("upper_ext_limit", 0),
            )

            if "attachment_addr" in kwargs:
                self.attachment_addr = kwargs["attachment_addr"]
                self.block_len += 8
                self.links_nr += 1
                self.attachment_nr = 1

        # ignore MLSD signal data
        if self.channel_type == v4c.CHANNEL_TYPE_MLSD:
            self.data_block_addr = 0
            self.channel_type = v4c.CHANNEL_TYPE_VALUE

        if self.name in self.display_names:
            del self.display_names[self.name]

        self.standard_C_size = True

    def __getitem__(self, item: str) -> Any:
        return self.__getattribute__(item)

    def __setitem__(self, item: str, value: Any) -> None:
        self.__setattr__(item, value)

    def to_blocks(
        self,
        address: int,
        blocks: list[Any],
        defined_texts: dict[str, int],
        cc_map: dict[bytes, int],
        si_map: dict[bytes, int],
    ) -> int:
        text = self.name
        if text in defined_texts:
            self.name_addr = defined_texts[text]
        else:
            tx_block = TextBlock(
                text=text.encode("utf-8", "replace"),
                meta=False,
                safe=True,
            )
            self.name_addr = defined_texts[text] = address
            tx_block.address = address
            address += tx_block.block_len
            blocks.append(tx_block)

        text = self.unit
        if text in defined_texts:
            self.unit_addr = defined_texts[text]
        else:
            tx_block = TextBlock(
                text=text.encode("utf-8", "replace"),
                meta=False,
                safe=True,
            )
            self.unit_addr = address
            defined_texts[text] = address
            tx_block.address = address
            address += tx_block.block_len
            blocks.append(tx_block)

        comment = self.comment
        display_names = self.display_names

        if display_names:
            items = []
            for _name, description in display_names.items():
                description = "display"
                items.append(f"<{description}>{_name}</{description}>")
            display_names_tags = "\n".join(items)
        else:
            display_names_tags = ""

        if display_names_tags and not comment:
            text = v4c.CN_COMMENT_TEMPLATE.format("", display_names_tags)

        elif display_names_tags and comment:
            if not comment.startswith("<CN"):
                text = v4c.CN_COMMENT_TEMPLATE.format(escape_xml_string(comment), display_names_tags)
            else:
                if any(_name not in comment for _name in display_names):
                    try:
                        CNcomment = ET.fromstring(comment)
                        text = CNcomment.find("TX").text
                        text = v4c.CN_COMMENT_TEMPLATE.format(escape_xml_string(text), display_names_tags)

                    except UnicodeEncodeError:
                        text = comment
                else:
                    text = comment
        else:
            text = comment

        if text in defined_texts:
            self.comment_addr = defined_texts[text]
        else:
            meta = text.startswith("<CN")
            tx_block = TextBlock(
                text=text.encode("utf-8", "replace"),
                meta=meta,
                safe=True,
            )
            self.comment_addr = address
            defined_texts[text] = address
            tx_block.address = address
            address += tx_block.block_len
            blocks.append(tx_block)

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
        # keep extra space for an eventual attachment address
        blocks.append(EIGHT_BYTES)
        self.address = address
        address += self.block_len + 8

        return address

    def __bytes__(self) -> bytes:
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
        elif self.attachment_nr == 1:
            return v4c.SINGLE_ATTACHMENT_CHANNEL_PACK(
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
                self.attachment_addr,
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

    def __str__(self) -> str:
        return f"""<Channel (name: {self.name}, unit: {self.unit}, comment: {self.comment}, address: {hex(self.address)},
    conversion: {self.conversion},
    source: {self.source},
    fields: {', '.join(block_fields(self))})>"""

    def metadata(self) -> str:
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
display names: {self.display_names}
address: {hex(self.address)}
comment: {self.comment}
unit: {self.unit}

""".split(
            "\n"
        )

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
            if key == "data_type":
                lines[-1] += f" = {v4c.DATA_TYPE_TO_STRING[self.data_type]}"
            elif key == "channel_type":
                lines[-1] += f" = {v4c.CHANNEL_TYPE_TO_STRING[self.channel_type]}"
            elif key == "sync_type":
                lines[-1] += f" = {v4c.SYNC_TYPE_TO_STRING[self.sync_type]}"
            elif key == "flags":
                flags = []
                for fl, string in v4c.FLAG_CN_TO_STRING.items():
                    if self.flags & fl:
                        flags.append(string)
                if flags:
                    lines[-1] += f" [0x{self.flags:X}= {', '.join(flags)}]"

        for line in lines:
            if not line:
                metadata.append(line)
            else:
                for wrapped_line in wrap(line, width=120):
                    metadata.append(wrapped_line)

        return "\n".join(metadata)

    def __contains__(self, item: str) -> bool:
        return hasattr(self, item)

    def __lt__(self, other: Channel) -> bool:
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
        "axis_channels",
        "dynamic_size_channels",
        "input_quantity_channels",
        "output_quantity_channel",
        "comparison_quantity_channel",
        "axis_conversions",
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
    * ``axis_channels`` - list : list of (group index, channel index)
      pairs referencing the axis of this array block
    * ``axis_conversions`` - list : list of ChannelConversion or None
      for each axis of this array block
    * ``dynamic_size_channels`` - list : list of (group index, channel index)
      pairs referencing the axis dynamic size of this array block
    * ``input_quantity_channels`` - list : list of (group index, channel index)
      pairs referencing the input quantity channels of this array block
    * ``output_quantity_channels`` - tuple | None : (group index, channel index)
      pair referencing the output quantity channel of this array block
    * ``comparison_quantity_channel`` - tuple | None : (group index, channel index)
      pair referencing the comparison quantity channel of this array block


    """

    def __init__(self, **kwargs) -> None:
        self.axis_channels = []
        self.dynamic_size_channels = []
        self.input_quantity_channels = []
        self.output_quantity_channel = None
        self.comparison_quantity_channel = None
        self.axis_conversions = []

        try:
            self.address = address = kwargs["address"]

            stream = kwargs["stream"]

            mapped = kwargs.get("mapped", False) or not is_file_like(stream)

            if mapped:
                (self.id, self.reserved0, self.block_len, self.links_nr) = COMMON_uf(stream, address)

                if self.id != b"##CA":
                    message = f'Expected "##CA" block @{hex(address)} but found "{self.id}"'
                    logger.exception(message)
                    raise MdfException(message)

                nr = self.links_nr
                address += COMMON_SIZE
                links = unpack_from(f"<{nr}Q", stream, address)
                self.composition_addr = links[0]
                links = links[1:]

                address += nr * 8
                values = unpack_from("<2BHIiI", stream, address)
                dims_nr = values[2]

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

                stream.seek(address + dims_nr * 8)

                if self.storage == v4c.CA_STORAGE_TYPE_DG_TEMPLATE:
                    data_links_nr = 1
                    for size in dim_sizes:
                        data_links_nr *= size

                    for i in range(data_links_nr):
                        self[f"data_link_{i}"] = links[i]

                    links = links[data_links_nr:]

                if self.flags & v4c.FLAG_CA_DYNAMIC_AXIS:
                    for i in range(dims_nr):
                        self[f"dynamic_size_{i}_dg_addr"] = links[3 * i]
                        self[f"dynamic_size_{i}_cg_addr"] = links[3 * i + 1]
                        self[f"dynamic_size_{i}_ch_addr"] = links[3 * i + 2]
                    links = links[dims_nr * 3 :]

                if self.flags & v4c.FLAG_CA_INPUT_QUANTITY:
                    for i in range(dims_nr):
                        self[f"input_quantity_{i}_dg_addr"] = links[3 * i]
                        self[f"input_quantity_{i}_cg_addr"] = links[3 * i + 1]
                        self[f"input_quantity_{i}_ch_addr"] = links[3 * i + 2]
                    links = links[dims_nr * 3 :]

                if self.flags & v4c.FLAG_CA_OUTPUT_QUANTITY:
                    self["output_quantity_dg_addr"] = links[0]
                    self["output_quantity_cg_addr"] = links[1]
                    self["output_quantity_ch_addr"] = links[2]
                    links = links[3:]

                if self.flags & v4c.FLAG_CA_COMPARISON_QUANTITY:
                    self["comparison_quantity_dg_addr"] = links[0]
                    self["comparison_quantity_cg_addr"] = links[1]
                    self["comparison_quantity_ch_addr"] = links[2]
                    links = links[3:]

                if self.flags & v4c.FLAG_CA_AXIS:
                    for i in range(dims_nr):
                        self[f"axis_conversion_{i}"] = links[i]
                    links = links[dims_nr:]

                if (self.flags & v4c.FLAG_CA_AXIS) and not (self.flags & v4c.FLAG_CA_FIXED_AXIS):
                    for i in range(dims_nr):
                        self[f"scale_axis_{i}_dg_addr"] = links[3 * i]
                        self[f"scale_axis_{i}_cg_addr"] = links[3 * i + 1]
                        self[f"scale_axis_{i}_ch_addr"] = links[3 * i + 2]

                    links = links[dims_nr * 3 :]

                if self.flags & v4c.FLAG_CA_FIXED_AXIS:
                    for i in range(dims_nr):
                        for j in range(self[f"dim_size_{i}"]):
                            (value,) = FLOAT64_u(stream.read(8))
                            self[f"axis_{i}_value_{j}"] = value
            else:
                stream.seek(address)

                (self.id, self.reserved0, self.block_len, self.links_nr) = unpack("<4sI2Q", stream.read(24))

                if self.id != b"##CA":
                    message = f'Expected "##CA" block @{hex(address)} but found "{self.id}"'
                    logger.exception(message)
                    raise MdfException(message)

                nr = self.links_nr
                links = unpack(f"<{nr}Q", stream.read(8 * nr))
                self.composition_addr = links[0]
                links = links[1:]

                values = unpack("<2BHIiI", stream.read(16))
                dims_nr = values[2]

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

                if self.storage == v4c.CA_STORAGE_TYPE_DG_TEMPLATE:
                    data_links_nr = 1
                    for size in dim_sizes:
                        data_links_nr *= size

                    for i in range(data_links_nr):
                        self[f"data_link_{i}"] = links[i]

                    links = links[data_links_nr:]

                if self.flags & v4c.FLAG_CA_DYNAMIC_AXIS:
                    for i in range(dims_nr):
                        self[f"dynamic_size_{i}_dg_addr"] = links[3 * i]
                        self[f"dynamic_size_{i}_cg_addr"] = links[3 * i + 1]
                        self[f"dynamic_size_{i}_ch_addr"] = links[3 * i + 2]
                    links = links[dims_nr * 3 :]

                if self.flags & v4c.FLAG_CA_INPUT_QUANTITY:
                    for i in range(dims_nr):
                        self[f"input_quantity_{i}_dg_addr"] = links[3 * i]
                        self[f"input_quantity_{i}_cg_addr"] = links[3 * i + 1]
                        self[f"input_quantity_{i}_ch_addr"] = links[3 * i + 2]
                    links = links[dims_nr * 3 :]

                if self.flags & v4c.FLAG_CA_OUTPUT_QUANTITY:
                    self["output_quantity_dg_addr"] = links[0]
                    self["output_quantity_cg_addr"] = links[1]
                    self["output_quantity_ch_addr"] = links[2]
                    links = links[3:]

                if self.flags & v4c.FLAG_CA_COMPARISON_QUANTITY:
                    self["comparison_quantity_dg_addr"] = links[0]
                    self["comparison_quantity_cg_addr"] = links[1]
                    self["comparison_quantity_ch_addr"] = links[2]
                    links = links[3:]

                if self.flags & v4c.FLAG_CA_AXIS:
                    for i in range(dims_nr):
                        self[f"axis_conversion_{i}"] = links[i]
                    links = links[dims_nr:]

                if (self.flags & v4c.FLAG_CA_AXIS) and not (self.flags & v4c.FLAG_CA_FIXED_AXIS):
                    for i in range(dims_nr):
                        self[f"scale_axis_{i}_dg_addr"] = links[3 * i]
                        self[f"scale_axis_{i}_cg_addr"] = links[3 * i + 1]
                        self[f"scale_axis_{i}_ch_addr"] = links[3 * i + 2]

                    links = links[dims_nr * 3 :]

                if self.flags & v4c.FLAG_CA_FIXED_AXIS:
                    for i in range(dims_nr):
                        for j in range(self[f"dim_size_{i}"]):
                            (value,) = FLOAT64_u(stream.read(8))
                            self[f"axis_{i}_value_{j}"] = value

        except KeyError:
            self.id = b"##CA"
            self.reserved0 = 0
            self.address = 0

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
                            self[f"axis_{i}_value_{j}"] = kwargs.get(f"axis_{i}_value_{j}", j)
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

    def __getitem__(self, item: str) -> Any:
        return self.__getattribute__(item)

    def __setitem__(self, item: str, value: Any) -> None:
        self.__setattr__(item, value)

    def __str__(self) -> str:
        return f"<ChannelArrayBlock (referenced channels: {self.axis_channels}, address: {hex(self.address)}, fields: {dict(self)})>"

    def __bytes__(self) -> bytes:
        flags = self.flags
        dims_nr = self.dims

        keys = (
            "id",
            "reserved0",
            "block_len",
            "links_nr",
            "composition_addr",
        )

        if self.storage:
            dim_sizes = [self[f"dim_size_{i}"] for i in range(dims_nr)]

            data_links_nr = 1
            for size in dim_sizes:
                data_links_nr *= size
        else:
            dim_sizes = []
            data_links_nr = 0

        if self.storage == v4c.CA_STORAGE_TYPE_DG_TEMPLATE:
            keys += tuple(f"data_link_{i}" for i in range(data_links_nr))

        if flags & v4c.FLAG_CA_DYNAMIC_AXIS:
            for i in range(dims_nr):
                keys += (
                    f"dynamic_size_{i}_dg_addr",
                    f"dynamic_size_{i}_cg_addr",
                    f"dynamic_size_{i}_ch_addr",
                )

        if flags & v4c.FLAG_CA_INPUT_QUANTITY:
            for i in range(dims_nr):
                keys += (
                    f"input_quantity_{i}_dg_addr",
                    f"input_quantity_{i}_cg_addr",
                    f"input_quantity_{i}_ch_addr",
                )

        if flags & v4c.FLAG_CA_OUTPUT_QUANTITY:
            keys += (
                "output_quantity_dg_addr",
                "output_quantity_cg_addr",
                "output_quantity_ch_addr",
            )

        if flags & v4c.FLAG_CA_COMPARISON_QUANTITY:
            keys += (
                "comparison_quantity_dg_addr",
                "comparison_quantity_cg_addr",
                "comparison_quantity_ch_addr",
            )

        if flags & v4c.FLAG_CA_AXIS:
            keys += tuple(f"axis_conversion_{i}" for i in range(dims_nr))

        if (flags & v4c.FLAG_CA_AXIS) and not (flags & v4c.FLAG_CA_FIXED_AXIS):
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

        if flags & v4c.FLAG_CA_FIXED_AXIS:
            keys += tuple(f"axis_{i}_value_{j}" for i in range(dims_nr) for j in range(self[f"dim_size_{i}"]))

            dim_sizes = [1 for i in range(dims_nr) for j in range(self[f"dim_size_{i}"])]

        if self.storage:
            keys += tuple(f"cycle_count_{i}" for i in range(data_links_nr))

        fmt = f"<4sI{self.links_nr + 2}Q2BHIiI{dims_nr}Q{sum(dim_sizes)}d{data_links_nr}Q"

        result = pack(fmt, *[getattr(self, key) for key in keys])
        return result

    def get_byte_offset_factors(self) -> list[int]:
        """Returns list of factors f(d), used to calculate byte offset"""
        return self._factors(self.byte_offset_base)

    def get_bit_pos_inval_factors(self) -> list[int]:
        """Returns list of factors f(d), used to calculate invalidation bit position"""
        return self._factors(self.invalidation_bit_base)

    def _factors(self, base: int) -> list[int]:
        factor = base
        factors = [factor]
        # column oriented layout
        if self.flags & v4c.FLAG_CA_INVERSE_LAYOUT:
            for i in range(1, self.dims):
                factor *= self[f"dim_size_{i - 1}"]
                factors.append(factor)

        # row oriented layout
        else:
            for i in range(self.dims - 2, -1, -1):
                factor *= self[f"dim_size_{i + 1}"]
                factors.insert(0, factor)

        return factors


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
    * ``acq_source_addr`` - int : address of SourceInformation that contains the
      channel group source
    * ``first_sample_reduction_addr`` - int : address of first SRBLOCK; this is
      considered 0 since sample reduction is not yet supported
    * ``comment_addr`` - int : address of TXBLOCK/MDBLOCK that contains the
      channel group comment
    * ``record_id`` - int : record ID for the channel group
    * ``cycles_nr`` - int : number of cycles for this channel group
    * ``flags`` - int : channel group flags
    * ``path_separator`` - int : ordinal for character used as path separator
    * ``reserved1`` - int : reserved bytes
    * ``samples_byte_nr`` - int : number of bytes used for channels samples in
      the record for this channel group; this does not contain the invalidation
      bytes
    * ``invalidation_bytes_nr`` - int : number of bytes used for invalidation
      bits by this channel group

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
        "cg_master_addr",
        "record_id",
        "cycles_nr",
        "flags",
        "path_separator",
        "reserved1",
        "samples_byte_nr",
        "invalidation_bytes_nr",
        "cg_master_index",
    )

    def __init__(self, **kwargs) -> None:
        self.acq_name = self.comment = ""
        self.acq_source = None
        self.cg_master_index = None

        try:
            self.address = address = kwargs["address"]
            stream = kwargs["stream"]
            mapped = kwargs.get("mapped", False) or not is_file_like(stream)

            if mapped:
                (self.id, self.reserved0, self.block_len, self.links_nr) = COMMON_uf(stream, address)

                if self.block_len == v4c.CG_BLOCK_SIZE:
                    (
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
                    ) = v4c.CHANNEL_GROUP_SHORT_uf(stream, address + COMMON_SIZE)

                else:
                    (
                        self.next_cg_addr,
                        self.first_ch_addr,
                        self.acq_name_addr,
                        self.acq_source_addr,
                        self.first_sample_reduction_addr,
                        self.comment_addr,
                        self.cg_master_addr,
                        self.record_id,
                        self.cycles_nr,
                        self.flags,
                        self.path_separator,
                        self.reserved1,
                        self.samples_byte_nr,
                        self.invalidation_bytes_nr,
                    ) = v4c.CHANNEL_GROUP_RM_SHORT_uf(stream, address + COMMON_SIZE)

            else:
                stream.seek(address)

                (
                    self.id,
                    self.reserved0,
                    self.block_len,
                    self.links_nr,
                ) = v4c.COMMON_u(stream.read(COMMON_SIZE))

                if self.block_len == v4c.CG_BLOCK_SIZE:
                    (
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
                    ) = v4c.CHANNEL_GROUP_SHORT_u(stream.read(v4c.CG_BLOCK_SIZE - COMMON_SIZE))

                else:
                    (
                        self.next_cg_addr,
                        self.first_ch_addr,
                        self.acq_name_addr,
                        self.acq_source_addr,
                        self.first_sample_reduction_addr,
                        self.comment_addr,
                        self.cg_master_addr,
                        self.record_id,
                        self.cycles_nr,
                        self.flags,
                        self.path_separator,
                        self.reserved1,
                        self.samples_byte_nr,
                        self.invalidation_bytes_nr,
                    ) = v4c.CHANNEL_GROUP_RM_SHORT_u(stream.read(v4c.CG_RM_BLOCK_SIZE - COMMON_SIZE))

            if self.id != b"##CG":
                message = f'Expected "##CG" block @{hex(address)} but found "{self.id}"'

                logger.exception(message)
                raise MdfException(message)

            self.acq_name = get_text_v4(self.acq_name_addr, stream, mapped=mapped)
            self.comment = get_text_v4(self.comment_addr, stream, mapped=mapped)

            si_map = kwargs["si_map"]

            address = self.acq_source_addr
            if address:
                if mapped:
                    raw_bytes = stream[address : address + v4c.SI_BLOCK_SIZE]
                else:
                    stream.seek(address)
                    raw_bytes = stream.read(v4c.SI_BLOCK_SIZE)
                if raw_bytes in si_map:
                    source = si_map[raw_bytes]
                else:
                    source = SourceInformation(
                        raw_bytes=raw_bytes,
                        stream=stream,
                        address=address,
                        mapped=mapped,
                        tx_map=kwargs["tx_map"],
                    )
                    si_map[raw_bytes] = source
                self.acq_source = source
            else:
                self.acq_source = None

        except KeyError:
            self.address = 0
            self.id = b"##CG"
            self.reserved0 = kwargs.get("reserved0", 0)

            self.next_cg_addr = kwargs.get("next_cg_addr", 0)
            self.first_ch_addr = kwargs.get("first_ch_addr", 0)
            self.acq_name_addr = kwargs.get("acq_name_addr", 0)
            self.acq_source_addr = kwargs.get("acq_source_addr", 0)
            self.first_sample_reduction_addr = kwargs.get("first_sample_reduction_addr", 0)
            self.comment_addr = kwargs.get("comment_addr", 0)
            self.record_id = kwargs.get("record_id", 1)
            self.cycles_nr = kwargs.get("cycles_nr", 0)
            self.flags = kwargs.get("flags", 0)
            self.path_separator = kwargs.get("path_separator", 0)
            self.reserved1 = kwargs.get("reserved1", 0)
            self.samples_byte_nr = kwargs.get("samples_byte_nr", 0)
            self.invalidation_bytes_nr = kwargs.get("invalidation_bytes_nr", 0)

            if self.flags & v4c.FLAG_CG_REMOTE_MASTER:
                self.cg_master_addr = kwargs.get("cg_master_addr", 0)
                self.block_len = v4c.CG_RM_BLOCK_SIZE
                self.links_nr = 7
            else:
                self.block_len = v4c.CG_BLOCK_SIZE
                self.links_nr = 6

    def __getitem__(self, item: str) -> Any:
        return self.__getattribute__(item)

    def __setitem__(self, item: str, value: Any) -> None:
        self.__setattr__(item, value)

    def to_blocks(
        self,
        address: int,
        blocks: list[Any],
        defined_texts: dict[str, int],
        si_map: dict[bytes, int],
    ) -> int:
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

    def __bytes__(self) -> bytes:
        if self.flags & v4c.FLAG_CG_REMOTE_MASTER:
            result = v4c.CHANNEL_GROUP_RM_p(
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
                self.cg_master_addr,
                self.record_id,
                self.cycles_nr,
                self.flags,
                self.path_separator,
                self.reserved1,
                self.samples_byte_nr,
                self.invalidation_bytes_nr,
            )
        else:
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

    def metadata(self) -> str:
        keys = (
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
        )
        if self.block_len == v4c.CG_RM_BLOCK_SIZE:
            keys += ("cg_master_addr",)

        keys += (
            "record_id",
            "cycles_nr",
            "flags",
            "path_separator",
            "reserved1",
            "samples_byte_nr",
            "invalidation_bytes_nr",
        )

        max_len = max(len(key) for key in keys)
        template = f"{{: <{max_len}}}: {{}}"

        metadata = []
        lines = f"""
name: {self.acq_name}
address: {hex(self.address)}
comment: {self.comment}

""".split(
            "\n"
        )

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

            if key == "flags":
                flags = []
                for fl, string in v4c.FLAG_CG_TO_STRING.items():
                    if self.flags & fl:
                        flags.append(string)
                if flags:
                    lines[-1] += f" [0x{self.flags:X}= {', '.join(flags)}]"
            elif key == "path_separator":
                if self.path_separator:
                    sep = pack("<H", self.path_separator).decode("utf-16")

                    lines[-1] += f" (= '{sep}')"
                else:
                    lines[-1] += " (= <undefined>)"

        for line in lines:
            if not line:
                metadata.append(line)
            else:
                for wrapped_line in wrap(line, width=120):
                    metadata.append(wrapped_line)

        return "\n".join(metadata)


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
    * ``inv_conv_addr`` int : address of inverse conversion
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

        * ``formula_addr`` - address of TXBLOCK that contains
          the algebraic conversion formula

    * tabular conversion with or without interpolation

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

    * text transformation (translation) conversion

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
    * ``referenced_blocks`` - dict : dict of referenced blocks; can be TextBlock
      objects for value to text, and text to text conversions; for partial
      conversions the referenced blocks can be ChannelConversion object as well
    * ``name`` - str : channel conversion name
    * ``unit`` - str : channel conversion unit

    """

    def __init__(self, **kwargs) -> None:
        self._cache = None
        self.is_user_defined = False

        if "stream" in kwargs:
            stream = kwargs["stream"]
            mapped = kwargs["mapped"]

            try:
                self.address = address = kwargs["address"]
                block = kwargs["raw_bytes"]
                (self.id, self.reserved0, self.block_len, self.links_nr) = COMMON_uf(block)

                if self.id != b"##CC":
                    message = f'Expected "##CC" block @{hex(address)} but found "{self.id}"'
                    logger.exception(message)
                    raise MdfException(message)

                block = block[COMMON_SIZE:]

            except KeyError:
                self.address = address = kwargs["address"]
                stream.seek(address)

                (self.id, self.reserved0, self.block_len, self.links_nr) = COMMON_u(stream.read(COMMON_SIZE))

                if self.id != b"##CC":
                    message = f'Expected "##CC" block @{hex(address)} but found "{self.id}"'
                    logger.exception(message)
                    raise MdfException(message) from None

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
                values = unpack_from(f"<{nr}d", block, 56)
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
                values = unpack_from(f"<{nr}d", block, 56)
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

                values = unpack_from(f"<{self.val_param_nr}d", block, 32 + links_nr * 8 + 24)
                self.default_lower = self.default_upper = 0
                for i in range(self.val_param_nr // 2):
                    j = 2 * i
                    self[f"lower_{i}"] = values[j]
                    self[f"upper_{i}"] = values[j + 1]

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

                values = unpack_from(f"<{self.val_param_nr}d", block, 32 + links_nr * 8 + 24)
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

            elif conv == v4c.CONVERSION_TYPE_BITFIELD:
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

                values = unpack_from(f"<{self.val_param_nr}Q", block, 32 + links_nr * 8 + 24)
                for i, val in enumerate(values):
                    self[f"mask_{i}"] = val

            self.referenced_blocks = None

            tx_map = kwargs["tx_map"]

            addr = self.name_addr
            if addr in tx_map:
                self.name = tx_map[addr]
            else:
                self.name = get_text_v4(addr, stream, mapped=mapped)
                tx_map[addr] = self.name

            addr = self.unit_addr
            if addr in tx_map:
                self.unit = tx_map[addr]
            else:
                self.unit = get_text_v4(addr, stream, mapped=mapped)
                tx_map[addr] = self.unit

            if isinstance(self.unit, bytes):
                self.unit = self.unit.decode("utf-8", errors="ignore")

            addr = self.comment_addr
            if addr in tx_map:
                self.comment = tx_map[addr]
            else:
                self.comment = get_text_v4(addr, stream, mapped=mapped)
                tx_map[addr] = self.comment

            conv_type = conv

            if conv_type == v4c.CONVERSION_TYPE_ALG:
                self.formula = get_text_v4(self.formula_addr, stream, mapped=mapped).replace("x", "X")
            else:
                self.formula = ""

                if conv_type in v4c.TABULAR_CONVERSIONS:
                    refs = self.referenced_blocks = {}
                    if conv_type in (
                        v4c.CONVERSION_TYPE_TTAB,
                        v4c.CONVERSION_TYPE_BITFIELD,
                    ):
                        tabs = self.links_nr - 4
                    else:
                        tabs = self.links_nr - 4 - 1
                    for i in range(tabs):
                        address = self[f"text_{i}"]
                        if address:
                            if address in tx_map:
                                txt = tx_map[address]
                                if not isinstance(txt, bytes):
                                    txt = txt.encode("utf-8")
                                refs[f"text_{i}"] = txt
                            else:
                                stream.seek(address)
                                _id = stream.read(4)

                                if _id == b"##TX":
                                    block = get_text_v4(
                                        address=address,
                                        stream=stream,
                                        mapped=mapped,
                                        decode=False,
                                    )
                                    tx_map[address] = block
                                    refs[f"text_{i}"] = block
                                elif _id == b"##CC":
                                    block = ChannelConversion(
                                        address=address,
                                        stream=stream,
                                        mapped=mapped,
                                        tx_map=tx_map,
                                    )
                                    refs[f"text_{i}"] = block
                                else:
                                    message = f'Expected "##TX" or "##CC" block @{hex(address)} but found "{_id}"'
                                    logger.exception(message)
                                    raise MdfException(message)

                        else:
                            refs[f"text_{i}"] = b""
                    if conv_type not in (
                        v4c.CONVERSION_TYPE_TTAB,
                        v4c.CONVERSION_TYPE_BITFIELD,
                    ):
                        address = self.default_addr
                        if address:
                            if address in tx_map:
                                txt = tx_map[address] or b""
                                if not isinstance(txt, bytes):
                                    txt = txt.encode("utf-8")
                                refs["default_addr"] = txt
                            else:
                                stream.seek(address)
                                _id = stream.read(4)

                                if _id == b"##TX":
                                    block = get_text_v4(
                                        address=address,
                                        stream=stream,
                                        mapped=mapped,
                                        decode=False,
                                    )
                                    tx_map[address] = block
                                    refs["default_addr"] = block
                                elif _id == b"##CC":
                                    block = ChannelConversion(
                                        address=address,
                                        stream=stream,
                                        mapped=mapped,
                                        tx_map=tx_map,
                                    )
                                    refs["default_addr"] = block
                                else:
                                    message = f'Expected "##TX" or "##CC" block @{hex(address)} but found "{_id}"'
                                    logger.exception(message)
                                    raise MdfException(message)
                        else:
                            refs["default_addr"] = b""

                elif conv_type == v4c.CONVERSION_TYPE_TRANS:
                    refs = self.referenced_blocks = {}
                    # link_nr - common links (4) - default text link (1)
                    for i in range((self.links_nr - 4 - 1) // 2):
                        for key in (f"input_{i}_addr", f"output_{i}_addr"):
                            address = self[key]

                            if address:
                                if address in tx_map:
                                    txt = tx_map[address] or b""
                                    if not isinstance(txt, bytes):
                                        txt = txt.encode("utf-8")
                                    refs[key] = txt
                                else:
                                    block = get_text_v4(
                                        address=address,
                                        stream=stream,
                                        mapped=mapped,
                                        decode=False,
                                    )
                                    tx_map[address] = block
                                    refs[key] = block
                            else:
                                refs[key] = b""
                    address = self.default_addr
                    if address:
                        if address in tx_map:
                            txt = tx_map[address] or b""
                            if not isinstance(txt, bytes):
                                txt = txt.encode("utf-8")
                            refs["default_addr"] = txt
                        else:
                            block = get_text_v4(
                                address=address,
                                stream=stream,
                                mapped=mapped,
                                decode=False,
                            )
                            refs["default_addr"] = block
                            tx_map[address] = block
                    else:
                        refs["default_addr"] = b""

        else:
            self.name = kwargs.get("name", "")
            self.unit = kwargs.get("unit", "")
            self.comment = kwargs.get("comment", "")
            self.formula = kwargs.get("formula", "")
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
                self.b = float(kwargs["b"])
                self.a = float(kwargs["a"])

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
                    self.referenced_blocks[key] = kwargs[key]
                self.default_addr = 0
                key = "default_addr"
                if "default_addr" in kwargs:
                    default = kwargs["default_addr"]
                else:
                    default = kwargs.get("default", b"")
                self.referenced_blocks[key] = default
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
                    self.referenced_blocks[key] = kwargs[key]
                self.default_addr = 0
                self.default_lower = self.default_upper = 0
                if "default_addr" in kwargs:
                    default = kwargs["default_addr"]
                else:
                    default = kwargs.get("default", b"")
                if isinstance(default, bytes) and b"{X}" in default:
                    default = default.decode("latin-1").replace("{X}", "X").split('"')[1]
                    default = ChannelConversion(conversion_type=v4c.CONVERSION_TYPE_ALG, formula=default)
                    self.referenced_blocks["default_addr"] = default
                else:
                    self.referenced_blocks["default_addr"] = default
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

            elif kwargs["conversion_type"] == v4c.CONVERSION_TYPE_BITFIELD:
                self.referenced_blocks = {}
                nr = kwargs["val_param_nr"]
                self.block_len = (nr * 8 * 2) + 80
                self.links_nr = nr + 4
                self.name_addr = kwargs.get("name_addr", 0)
                self.unit_addr = kwargs.get("unit_addr", 0)
                self.comment_addr = kwargs.get("comment_addr", 0)
                self.inv_conv_addr = kwargs.get("inv_conv_addr", 0)
                for i in range(nr):
                    key = f"text_{i}"
                    self[key] = 0
                    self.referenced_blocks[key] = kwargs[key]

                self.conversion_type = v4c.CONVERSION_TYPE_BITFIELD
                self.precision = kwargs.get("precision", 0)
                self.flags = kwargs.get("flags", 0)
                self.ref_param_nr = nr
                self.val_param_nr = nr
                self.min_phy_value = kwargs.get("min_phy_value", 0)
                self.max_phy_value = kwargs.get("max_phy_value", 0)
                for i in range(nr):
                    self[f"mask_{i}"] = kwargs[f"mask_{i}"]

            else:
                message = "Conversion {} dynamic creation not implemented"
                message = message.format(kwargs["conversion_type"])
                logger.exception(message)
                raise MdfException(message)

    def to_blocks(
        self,
        address: int,
        blocks: list[Any],
        defined_texts: dict[str, int],
        cc_map: dict[bytes, int],
    ) -> int:
        if id(self) in cc_map:
            return address

        text = self.name
        if text:
            if text in defined_texts:
                self.name_addr = defined_texts[text]
            else:
                tx_block = TextBlock(
                    text=text.encode("utf-8", "replace"),
                    meta=False,
                    safe=True,
                )
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
                tx_block = TextBlock(
                    text=text.encode("utf-8", "replace"),
                    meta=False,
                    safe=True,
                )
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
                    tx_block = TextBlock(
                        text=text.encode("utf-8", "replace"),
                        meta=False,
                        safe=True,
                    )
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
                tx_block = TextBlock(
                    text=text.encode("utf-8", "replace"),
                    meta=meta,
                    safe=True,
                )
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
                    if isinstance(block, ChannelConversion):
                        address = block.to_blocks(address, blocks, defined_texts, cc_map)
                        self[key] = block.address
                    else:
                        text = block
                        if text in defined_texts:
                            self[key] = defined_texts[text]
                        else:
                            block = TextBlock(text=text)
                            defined_texts[text] = address
                            blocks.append(block)
                            self[key] = address
                            address += block["block_len"]

                else:
                    self[key] = 0

        bts = bytes(self)
        if bts in cc_map:
            self.address = cc_map[bts]
        else:
            blocks.append(bts)
            self.address = address
            cc_map[bts] = cc_map[id(self)] = address
            address += self.block_len

        return address

    def convert(self, values, as_object=False, as_bytes=False, ignore_value2text_conversions=False):
        if not isinstance(values, np.ndarray):
            values = np.array(values)

        values_count = len(values)

        conversion_type = self.conversion_type
        if conversion_type == v4c.CONVERSION_TYPE_NON:
            pass
        elif conversion_type == v4c.CONVERSION_TYPE_LIN:
            a = self.a
            b = self.b

            if (a, b) != (1, 0):
                if values.dtype.names:
                    names = values.dtype.names
                    name = names[0]
                    vals = values[name]

                    vals = vals * a
                    if b:
                        vals += b
                    values = np.rec.fromarrays(
                        [vals] + [values[name] for name in names[1:]],
                        dtype=[(name, vals.dtype, vals.shape[1:])]
                        + [(name, values[name].dtype, values[name].shape[1:]) for name in names[1:]],
                    )

                else:
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

            names = values.dtype.names
            if names:
                name = names[0]
                vals = values[name]
                if (P1, P3, P4, P5) == (0, 0, 0, 0):
                    if P2 != P6:
                        vals = vals * (P2 / P6)

                elif (P2, P3, P4, P6) == (0, 0, 0, 0):
                    if P1 != P5:
                        vals = vals * (P1 / P5)

                else:
                    X = vals
                    try:
                        vals = evaluate(v4c.CONV_RAT_TEXT)
                    except TypeError:
                        vals = (P1 * X**2 + P2 * X + P3) / (P4 * X**2 + P5 * X + P6)

                values = np.rec.fromarrays(
                    [vals] + [values[name] for name in names[1:]],
                    dtype=[(name, vals.dtype, vals.shape[1:])]
                    + [(name, values[name].dtype, values[name].shape[1:]) for name in names[1:]],
                )

            else:
                X = values
                if (P1, P3, P4, P5) == (0, 0, 0, 0):
                    if P2 != P6:
                        values = values * (P2 / P6)

                elif (P2, P3, P4, P6) == (0, 0, 0, 0):
                    if P1 != P5:
                        values = values * (P1 / P5)
                else:
                    try:
                        values = evaluate(v4c.CONV_RAT_TEXT)
                    except TypeError:
                        values = (P1 * X**2 + P2 * X + P3) / (P4 * X**2 + P5 * X + P6)

        elif conversion_type == v4c.CONVERSION_TYPE_ALG:
            try:
                X = values
                INF = np.inf
                NaN = np.nan
                values = evaluate(self.formula.replace("X1", "X"))
            except:
                if lambdify is not None:
                    X_symbol = symbols("X")
                    expr = lambdify(
                        X_symbol,
                        self.formula.replace("X1", "X"),
                        modules=[{"INF": np.inf, "NaN": np.nan}, "numpy"],
                        dummify=False,
                        cse=True,
                    )
                    values = expr(values)

        elif conversion_type in (v4c.CONVERSION_TYPE_TABI, v4c.CONVERSION_TYPE_TAB):
            nr = self.val_param_nr // 2
            raw_vals = np.array([self[f"raw_{i}"] for i in range(nr)])
            phys = np.array([self[f"phys_{i}"] for i in range(nr)])

            if conversion_type == v4c.CONVERSION_TYPE_TABI:
                values = np.interp(values, raw_vals, phys)
            else:
                dim = raw_vals.shape[0]

                inds = np.searchsorted(raw_vals, values)

                inds[inds >= dim] = dim - 1

                inds2 = inds - 1
                inds2[inds2 < 0] = 0

                cond = np.abs(values - raw_vals[inds]) >= np.abs(values - raw_vals[inds2])

                values = np.where(cond, phys[inds2], phys[inds])

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

        elif conversion_type == v4c.CONVERSION_TYPE_TABX and values_count >= 150:
            if ignore_value2text_conversions:
                nr = self.val_param_nr
                raw_vals = [self[f"val_{i}"] for i in range(nr)]

                identical = ChannelConversion(conversion_type=v4c.CONVERSION_TYPE_NON)

                phys = []
                for i in range(nr):
                    ref = self.referenced_blocks[f"text_{i}"]
                    if isinstance(ref, bytes):
                        phys.append(identical)
                    else:
                        phys.append(ref)

                x = sorted(zip(raw_vals, phys))
                raw_vals = np.array([e[0] for e in x], dtype="<i8")
                phys = [e[1] for e in x]

                ref = self.referenced_blocks["default_addr"]
                if isinstance(ref, bytes):
                    default = identical
                else:
                    default = ref

            else:
                if self._cache is None or self._cache["type"] != "big":
                    nr = self.val_param_nr
                    raw_vals = [self[f"val_{i}"] for i in range(nr)]

                    phys = [self.referenced_blocks[f"text_{i}"] for i in range(nr)]

                    x = sorted(zip(raw_vals, phys))
                    raw_vals = np.array([e[0] for e in x], dtype="<i8")
                    phys = [e[1] for e in x]

                    self._cache = {"phys": phys, "raw_vals": raw_vals, "type": "big"}
                else:
                    phys = self._cache["phys"]
                    raw_vals = self._cache["raw_vals"]

                default = self.referenced_blocks["default_addr"]

            names = values.dtype.names

            if names:
                name = names[0]
                vals = values[name]
                shape = vals.shape
                vals = vals.ravel()

                ret = np.full(len(vals), None, "O")

                idx1 = np.searchsorted(raw_vals, vals, side="right") - 1
                idx2 = np.searchsorted(raw_vals, vals, side="left")

                idx = np.argwhere(idx1 != idx2).ravel()

                if isinstance(default, bytes):
                    ret[idx] = default
                else:
                    ret[idx] = default.convert(vals[idx], ignore_value2text_conversions=ignore_value2text_conversions)

                idx = np.argwhere(idx1 == idx2).ravel()
                if idx.size:
                    indexes = idx1[idx]
                    unique = np.unique(indexes)
                    for val in unique.tolist():
                        item = phys[val]
                        idx_ = np.argwhere(indexes == val).ravel()
                        if isinstance(item, bytes):
                            ret[idx[idx_]] = item
                        else:
                            ret[idx[idx_]] = item.convert(
                                vals[idx[idx_]], ignore_value2text_conversions=ignore_value2text_conversions
                            )

                all_bytes = True
                for v in ret.tolist():
                    if not isinstance(v, bytes):
                        all_bytes = False
                        break

                if not all_bytes:
                    try:
                        ret = ret.astype("f8")
                    except:
                        if as_bytes:
                            ret = ret.astype(bytes)
                        elif not as_object:
                            ret = np.array([np.nan if isinstance(v, bytes) else v for v in ret.tolist()])

                else:
                    ret = ret.astype(bytes)

                ret = ret.reshape(shape)
                values = np.rec.fromarrays(
                    [ret] + [values[name] for name in names[1:]],
                    dtype=[(name, ret.dtype, ret.shape[1:])]
                    + [(name, values[name].dtype, values[name].shape[1:]) for name in names[1:]],
                )
            else:
                ret = np.full(values.size, None, "O")

                idx1 = np.searchsorted(raw_vals, values, side="right") - 1
                idx2 = np.searchsorted(raw_vals, values, side="left")

                idx = np.argwhere(idx1 != idx2).ravel()

                if idx.size:
                    # some raw values were not found in the conversion table
                    # so the default physical value must be returned

                    if isinstance(default, bytes):
                        ret[idx] = default
                    else:
                        ret[idx] = default.convert(
                            values[idx], ignore_value2text_conversions=ignore_value2text_conversions
                        )

                    idx = np.argwhere(idx1 == idx2).ravel()

                    if idx.size:
                        indexes = idx1[idx]

                        if indexes.size <= 300:
                            unique = sorted(set(indexes.tolist()))
                        else:
                            unique = np.unique(indexes).tolist()
                        for val in unique:
                            item = phys[val]
                            idx_ = np.argwhere(indexes == val).ravel()
                            if isinstance(item, bytes):
                                ret[idx[idx_]] = item
                            else:
                                ret[idx[idx_]] = item.convert(
                                    values[idx[idx_]], ignore_value2text_conversions=ignore_value2text_conversions
                                )

                else:
                    # all the raw values are found in the conversion table

                    if idx1.size:
                        if idx1.size <= 300:
                            unique = sorted(set(idx1.tolist()))
                        else:
                            unique = np.unique(idx1).tolist()
                        for val in unique:
                            item = phys[val]
                            idx_ = np.argwhere(idx1 == val).ravel()
                            if isinstance(item, bytes):
                                ret[idx_] = item
                            else:
                                ret[idx_] = item.convert(
                                    values[idx_], ignore_value2text_conversions=ignore_value2text_conversions
                                )

                all_bytes = True
                for v in ret.tolist():
                    if not isinstance(v, bytes):
                        all_bytes = False
                        break

                if not all_bytes:
                    try:
                        ret = ret.astype("f8")
                    except:
                        if as_bytes:
                            ret = ret.astype(bytes)
                        elif not as_object:
                            ret = np.array([np.nan if isinstance(v, bytes) else v for v in ret.tolist()])
                else:
                    ret = ret.astype(bytes)

                values = ret

        elif conversion_type == v4c.CONVERSION_TYPE_TABX:
            if ignore_value2text_conversions:
                nr = self.val_param_nr
                raw_vals = [self[f"val_{i}"] for i in range(nr)]

                identical = ChannelConversion(conversion_type=v4c.CONVERSION_TYPE_NON)

                phys = []
                for i in range(nr):
                    ref = self.referenced_blocks[f"text_{i}"]
                    if isinstance(ref, bytes):
                        phys.append(identical)
                    else:
                        phys.append(ref)

                x = sorted(zip(raw_vals, phys))
                raw_vals = [e[0] for e in x]
                phys = [e[1] for e in x]

                ref = self.referenced_blocks["default_addr"]
                if isinstance(ref, bytes):
                    default = identical
                else:
                    default = ref

                default_is_bytes = False

            else:
                if self._cache is None or self._cache["type"] != "small":
                    nr = self.val_param_nr
                    raw_vals = [self[f"val_{i}"] for i in range(nr)]

                    phys = [self.referenced_blocks[f"text_{i}"] for i in range(nr)]

                    x = sorted(zip(raw_vals, phys))
                    raw_vals = [e[0] for e in x]
                    phys = [e[1] for e in x]

                    self._cache = {"phys": phys, "raw_vals": raw_vals, "type": "small"}
                else:
                    phys = self._cache["phys"]
                    raw_vals = self._cache["raw_vals"]

                default = self.referenced_blocks["default_addr"]
                default_is_bytes = isinstance(default, bytes)

            names = values.dtype.names

            if names:
                name = names[0]
                vals = values[name]
                shape = vals.shape
                vals = vals.ravel().tolist()

                ret = []
                all_bytes = True

                for v in vals:
                    try:
                        v_ = phys[raw_vals.index(v)]
                        if isinstance(v_, bytes):
                            ret.append(v_)
                        else:
                            v_ = v_.convert([v], ignore_value2text_conversions=ignore_value2text_conversions)[0]
                            ret.append(v_)
                            if all_bytes and not isinstance(v_, bytes):
                                all_bytes = False

                    except:
                        if default_is_bytes:
                            ret.append(default)

                        else:
                            v_ = default.convert([v], ignore_value2text_conversions=ignore_value2text_conversions)[0]
                            ret.append(v_)
                            if all_bytes and not isinstance(v_, bytes):
                                all_bytes = False

                if not all_bytes:
                    try:
                        ret = np.array(ret, dtype="<f8")
                    except:
                        ret = np.array(ret, dtype="O")
                        if as_bytes:
                            ret = ret.astype(bytes)
                        elif not as_object:
                            ret = np.array([np.nan if isinstance(v, bytes) else v for v in ret])

                else:
                    ret = np.array(ret, dtype=bytes)

                ret = ret.reshape(shape)
                values = np.rec.fromarrays(
                    [ret] + [values[name] for name in names[1:]],
                    dtype=[(name, ret.dtype, ret.shape[1:])]
                    + [(name, values[name].dtype, values[name].shape[1:]) for name in names[1:]],
                )
            else:
                ret = []
                all_bytes = True
                vals = values.tolist()

                for v in vals:
                    try:
                        v_ = phys[raw_vals.index(v)]
                        if isinstance(v_, bytes):
                            ret.append(v_)
                        else:
                            v_ = v_.convert([v], ignore_value2text_conversions=ignore_value2text_conversions)[0]
                            ret.append(v_)
                            if all_bytes and not isinstance(v_, bytes):
                                all_bytes = False

                    except:
                        if default_is_bytes:
                            ret.append(default)

                        else:
                            v_ = default.convert([v], ignore_value2text_conversions=ignore_value2text_conversions)[0]
                            ret.append(v_)
                            if all_bytes and not isinstance(v_, bytes):
                                all_bytes = False

                if not all_bytes:
                    try:
                        ret = np.array(ret, dtype="<f8")
                    except:
                        ret = np.array(ret, dtype="O")
                        if as_bytes:
                            ret = ret.astype(bytes)
                        elif not as_object:
                            ret = np.array([np.nan if isinstance(v, bytes) else v for v in ret.tolist()])

                else:
                    ret = np.array(ret, dtype=bytes)

                values = ret

        elif conversion_type == v4c.CONVERSION_TYPE_RTABX and values_count >= 100:
            if ignore_value2text_conversions:
                nr = self.val_param_nr // 2

                identical = ChannelConversion(conversion_type=v4c.CONVERSION_TYPE_NON)

                phys = []
                for i in range(nr):
                    ref = self.referenced_blocks[f"text_{i}"]
                    if isinstance(ref, bytes):
                        phys.append(identical)
                    else:
                        phys.append(ref)

                lower = [self[f"lower_{i}"] for i in range(nr)]
                upper = [self[f"upper_{i}"] for i in range(nr)]

                x = sorted(zip(lower, upper, phys))
                lower = np.array([e[0] for e in x], dtype="<i8")
                upper = np.array([e[1] for e in x], dtype="<i8")
                phys = [e[2] for e in x]

                ref = self.referenced_blocks["default_addr"]
                if isinstance(ref, bytes):
                    default = identical
                else:
                    default = ref

                default_is_bytes = False

            else:
                if self._cache is None or self._cache["type"] != "big":
                    nr = self.val_param_nr // 2

                    phys = [self.referenced_blocks[f"text_{i}"] for i in range(nr)]

                    lower = [self[f"lower_{i}"] for i in range(nr)]
                    upper = [self[f"upper_{i}"] for i in range(nr)]

                    x = sorted(zip(lower, upper, phys))
                    lower = np.array([e[0] for e in x], dtype="<i8")
                    upper = np.array([e[1] for e in x], dtype="<i8")
                    phys = [e[2] for e in x]

                    self._cache = {
                        "phys": phys,
                        "lower": lower,
                        "upper": upper,
                        "type": "big",
                    }
                else:
                    phys = self._cache["phys"]
                    lower = self._cache["lower"]
                    upper = self._cache["upper"]

                default = self.referenced_blocks["default_addr"]

            ret = np.full(values.size, None, "O")

            idx1 = np.searchsorted(lower, values, side="right") - 1
            idx2 = np.searchsorted(upper, values, side="left")

            idx_ne = np.argwhere(idx1 != idx2).ravel()
            idx_eq = np.argwhere(idx1 == idx2).ravel()

            if isinstance(default, bytes):
                ret[idx_ne] = default
            else:
                ret[idx_ne] = default.convert(
                    values[idx_ne], ignore_value2text_conversions=ignore_value2text_conversions
                )

            if idx_eq.size:
                indexes = idx1[idx_eq]
                unique = np.unique(indexes)
                for val in unique:
                    item = phys[val]
                    idx_ = np.argwhere(indexes == val).ravel()

                    if isinstance(item, bytes):
                        ret[idx_eq[idx_]] = item
                    else:
                        try:
                            ret[idx_eq[idx_]] = item.convert(
                                values[idx_eq[idx_]], ignore_value2text_conversions=ignore_value2text_conversions
                            )
                        except:
                            raise

            all_bytes = True
            for v in ret.tolist():
                if not isinstance(v, bytes):
                    all_bytes = False
                    break

            if not all_bytes:
                try:
                    ret = ret.astype("f8")
                except:
                    if as_bytes:
                        ret = ret.astype(bytes)
                    elif not as_object:
                        ret = np.array([np.nan if isinstance(v, bytes) else v for v in ret.tolist()])
            else:
                ret = ret.astype(bytes)

            values = ret

        elif conversion_type == v4c.CONVERSION_TYPE_RTABX:
            if ignore_value2text_conversions:
                nr = self.val_param_nr // 2

                identical = ChannelConversion(conversion_type=v4c.CONVERSION_TYPE_NON)

                phys = []
                for i in range(nr):
                    ref = self.referenced_blocks[f"text_{i}"]
                    if isinstance(ref, bytes):
                        phys.append(identical)
                    else:
                        phys.append(ref)

                lower = [self[f"lower_{i}"] for i in range(nr)]
                upper = [self[f"upper_{i}"] for i in range(nr)]

                x = sorted(zip(lower, upper, phys))
                lower = [e[0] for e in x]
                upper = [e[1] for e in x]
                phys = [e[2] for e in x]

                ref = self.referenced_blocks["default_addr"]
                if isinstance(ref, bytes):
                    default = identical
                else:
                    default = ref

                default_is_bytes = False

            else:
                if self._cache is None or self._cache["type"] != "small":
                    nr = self.val_param_nr // 2

                    phys = [self.referenced_blocks[f"text_{i}"] for i in range(nr)]

                    lower = [self[f"lower_{i}"] for i in range(nr)]
                    upper = [self[f"upper_{i}"] for i in range(nr)]

                    x = sorted(zip(lower, upper, phys))
                    lower = [e[0] for e in x]
                    upper = [e[1] for e in x]
                    phys = [e[2] for e in x]

                    self._cache = {
                        "phys": phys,
                        "lower": lower,
                        "upper": upper,
                        "type": "small",
                    }
                else:
                    phys = self._cache["phys"]
                    lower = self._cache["lower"]
                    upper = self._cache["upper"]

                default = self.referenced_blocks["default_addr"]
                default_is_bytes = isinstance(default, bytes)

            ret = []
            vals = values.tolist()
            all_bytes = True

            if values.dtype.kind in "ui":
                for v in vals:
                    for l, u, p in zip(lower, upper, phys):
                        if l <= v <= u:
                            if isinstance(p, bytes):
                                ret.append(p)
                            else:
                                p = p.convert([v], ignore_value2text_conversions=ignore_value2text_conversions)[0]
                                ret.append(p)
                                if all_bytes and not isinstance(p, bytes):
                                    all_bytes = False
                            break
                    else:
                        if default_is_bytes:
                            ret.append(default)

                        else:
                            v_ = default.convert([v], ignore_value2text_conversions=ignore_value2text_conversions)[0]
                            ret.append(v_)
                            if all_bytes and not isinstance(v_, bytes):
                                all_bytes = False

            else:
                for v in vals:
                    for l, u, p in zip(lower, upper, phys):
                        if l <= v < u:
                            if isinstance(p, bytes):
                                ret.append(p)
                            else:
                                p = p.convert([v], ignore_value2text_conversions=ignore_value2text_conversions)[0]
                                ret.append(p)
                                if all_bytes and not isinstance(p, bytes):
                                    all_bytes = False
                            break
                    else:
                        if default_is_bytes:
                            ret.append(default)

                        else:
                            v_ = default.convert([v], ignore_value2text_conversions=ignore_value2text_conversions)[0]
                            ret.append(v_)
                            if all_bytes and not isinstance(v_, bytes):
                                all_bytes = False

            if not all_bytes:
                try:
                    ret = np.array(ret, dtype="f8")
                except:
                    ret = np.array(ret, dtype="O")
                    if as_bytes:
                        ret = ret.astype(bytes)
                    elif not as_object:
                        ret = np.array([np.nan if isinstance(v, bytes) else v for v in ret.tolist()])
            else:
                ret = np.array(ret, dtype=bytes)

            values = ret

        elif conversion_type == v4c.CONVERSION_TYPE_TTAB:
            nr = self.val_param_nr - 1

            raw_values = [self.referenced_blocks[f"text_{i}"] for i in range(nr)]
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
            if not ignore_value2text_conversions:
                nr = (self.ref_param_nr - 1) // 2

                in_ = [self.referenced_blocks[f"input_{i}_addr"] for i in range(nr)]

                out_ = [self.referenced_blocks[f"output_{i}_addr"] for i in range(nr)]
                default = self.referenced_blocks["default_addr"]

                new_values = []
                for val in values:
                    try:
                        val = out_[in_.index(val.strip(b"\0"))]
                    except ValueError:
                        val = default
                    new_values.append(val)

                values = np.array(new_values)

        elif conversion_type == v4c.CONVERSION_TYPE_BITFIELD:
            if not ignore_value2text_conversions:
                nr = self.val_param_nr

                phys = [self.referenced_blocks[f"text_{i}"] for i in range(nr)]
                masks = np.array([self[f"mask_{i}"] for i in range(nr)], dtype="u8")

                phys = [
                    (
                        conv
                        if isinstance(conv, bytes)
                        else ((f"{conv.name}=".encode(), conv) if conv.name else (b"", conv))
                    )
                    for conv in phys
                ]

                new_values = []
                values = values.astype("u8").tolist()
                for val in values:
                    new_val = []
                    masked_values = (masks & val).tolist()

                    for on, conv in zip(masked_values, phys):
                        if not on:
                            continue

                        if isinstance(conv, bytes):
                            if conv:
                                new_val.append(conv)
                        else:
                            prefix, conv = conv
                            converted_val = conv.convert(
                                [on], ignore_value2text_conversions=ignore_value2text_conversions
                            )
                            if converted_val:
                                if prefix:
                                    new_val.append(prefix + converted_val)
                                else:
                                    new_val.append(converted_val)

                    new_values.append(b"|".join(new_val))

                values = np.array(new_values)

        return values

    def metadata(self, indent: str = "") -> str:
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

        elif self.conversion_type == v4c.CONVERSION_TYPE_BITFIELD:
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
            keys += tuple(f"mask_{i}" for i in range(self.val_param_nr))

        max_len = max(len(key) for key in keys)
        template = f"{{: <{max_len}}}: {{}}"

        metadata = []
        lines = f"""
name: {self.name}
unit: {self.unit}
address: {hex(self.address)}
comment: {self.comment}
formula: {self.formula}

""".split(
            "\n"
        )
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

            if key == "conversion_type":
                lines[-1] += f" = {v4c.CONVERSION_TYPE_TO_STRING[self.conversion_type]}"
            elif self.referenced_blocks and key in self.referenced_blocks:
                val = self.referenced_blocks[key]
                if isinstance(val, bytes):
                    lines[-1] += f" (= {str(val)[1:]})"
                else:
                    lines[-1] += f" (= CCBLOCK @ {hex(val.address)})"

        if self.referenced_blocks:
            max_len = max(len(key) for key in self.referenced_blocks)
            template = f"{{: <{max_len}}}: {{}}"

            lines.append("")
            lines.append("Referenced blocks:")
            for key, block in self.referenced_blocks.items():
                if isinstance(block, ChannelConversion):
                    lines.append(template.format(key, ""))
                    lines.extend(block.metadata(indent + "    ").split("\n"))
                else:
                    lines.append(template.format(key, str(block)[1:]))

        for line in lines:
            if not line:
                metadata.append(line)
            else:
                for wrapped_line in wrap(line, initial_indent=indent, subsequent_indent=indent, width=120):
                    metadata.append(wrapped_line)

        return "\n".join(metadata)

    def __getitem__(self, item: str) -> Any:
        return self.__getattribute__(item)

    def __setitem__(self, item: str, value: Any) -> None:
        self.__setattr__(item, value)

    def __contains__(self, item: str) -> bool:
        return hasattr(self, item)

    def __bytes__(self) -> bytes:
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
            fmt = f"<4sI{self.links_nr + 2}Q2B3H{self.val_param_nr + 2}d"
            keys = v4c.KEYS_CONVERSION_NONE
            for i in range(self.val_param_nr // 2):
                keys += (f"raw_{i}", f"phys_{i}")
            result = pack(fmt, *[getattr(self, key) for key in keys])
        elif self.conversion_type == v4c.CONVERSION_TYPE_RTAB:
            fmt = f"<4sI{self.links_nr + 2}Q2B3H{self.val_param_nr + 2}d"
            keys = v4c.KEYS_CONVERSION_NONE
            for i in range(self.val_param_nr // 3):
                keys += (f"lower_{i}", f"upper_{i}", f"phys_{i}")
            keys += ("default",)
            result = pack(fmt, *[getattr(self, key) for key in keys])
        elif self.conversion_type == v4c.CONVERSION_TYPE_TABX:
            fmt = f"<4sI{self.links_nr + 2}Q2B3H{self.val_param_nr + 2}d"
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
            fmt = f"<4sI{self.links_nr + 2}Q2B3H{self.val_param_nr + 2}d"
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
            result = pack(fmt, *[getattr(self, key) for key in keys])
        elif self.conversion_type == v4c.CONVERSION_TYPE_TTAB:
            fmt = f"<4sI{self.links_nr + 2}Q2B3H{self.val_param_nr + 2}d"
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
            fmt = f"<4sI{self.links_nr + 2}Q2B3H{self.val_param_nr + 2}d"
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

        elif self.conversion_type == v4c.CONVERSION_TYPE_BITFIELD:
            fmt = f"<4sI{self.links_nr + 2}Q2B3H2d{self.val_param_nr}Q"
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
            keys += tuple(f"text_{i}" for i in range(self.val_param_nr))
            keys += (
                "conversion_type",
                "precision",
                "flags",
                "ref_param_nr",
                "val_param_nr",
                "min_phy_value",
                "max_phy_value",
            )
            keys += tuple(f"mask_{i}" for i in range(self.val_param_nr))
            result = pack(fmt, *[getattr(self, key) for key in keys])

        return result

    def __str__(self) -> str:
        return f"<ChannelConversion (name: {self.name}, unit: {self.unit}, comment: {self.comment}, formula: {self.formula}, referenced blocks: {self.referenced_blocks}, address: {self.address}, fields: {block_fields(self)})>"


class DataBlock:
    """Common implementation for DTBLOCK/RDBLOCK/SDBLOCK/DVBLOCK/DIBLOCK

    *DataBlock* has the following attributes, that are also available as
    dict like key-value pairs

    DTBLOCK fields

    * ``id`` - bytes : block ID; b'##DT' for DTBLOCK, b'##RD' for RDBLOCK,
      b'##SD' for SDBLOCK, b'##DV' for DVBLOCK or b'##DI' for DIBLOCK
    * ``reserved0`` - int : reserved bytes
    * ``block_len`` - int : block bytes size
    * ``links_nr`` - int : number of links
    * ``data`` - bytes : raw samples

    Other attributes

    * ``address`` - int : data block address

    Parameters
    ----------
    address : int
        DTBLOCK/RDBLOCK/SDBLOCK/DVBLOCK/DIBLOCK address inside the file
    stream : int
        file handle
    reduction : bool
        sample reduction data block

    """

    __slots__ = ("address", "id", "reserved0", "block_len", "links_nr", "data")

    def __init__(self, **kwargs) -> None:
        try:
            self.address = address = kwargs["address"]
            stream = kwargs["stream"]
            mapped = kwargs.get("mapped", False) or not is_file_like(stream)

            if mapped:
                (self.id, self.reserved0, self.block_len, self.links_nr) = COMMON_uf(stream, address)

                if self.id not in (b"##DT", b"##RD", b"##SD", b"##DV", b"##DI"):
                    message = (
                        f'Expected "##DT", "##DV", "##DI", "##RD" or "##SD" block @{hex(address)} but found "{self.id}"'
                    )
                    logger.exception(message)
                    raise MdfException(message)

                self.data = stream[address + COMMON_SIZE : address + self.block_len]
            else:
                stream.seek(address)

                (self.id, self.reserved0, self.block_len, self.links_nr) = COMMON_u(stream.read(COMMON_SIZE))

                if self.id not in (b"##DT", b"##RD", b"##SD", b"##DV", b"##DI"):
                    message = (
                        f'Expected "##DT", "##DV", "##DI", "##RD" or "##SD" block @{hex(address)} but found "{self.id}"'
                    )
                    logger.exception(message)
                    raise MdfException(message)

                self.data = stream.read(self.block_len - COMMON_SIZE)

        except KeyError:
            self.address = 0
            type = kwargs.get("type", "DT")
            if type not in ("DT", "SD", "RD", "DV", "DI"):
                type = "DT"

            self.id = f"##{type}".encode("ascii")
            self.reserved0 = 0
            self.block_len = len(kwargs["data"]) + COMMON_SIZE
            self.links_nr = 0
            self.data = kwargs["data"]

    def __getitem__(self, item: str) -> Any:
        return self.__getattribute__(item)

    def __setitem__(self, item: str, value: Any) -> None:
        self.__setattr__(item, value)

    def __bytes__(self) -> bytes:
        return v4c.COMMON_p(self.id, self.reserved0, self.block_len, self.links_nr) + self.data


class DataZippedBlock:
    """*DataZippedBlock* has the following attributes, that are also available
    as dict like key-value pairs

    DZBLOCK fields

    * ``id`` - bytes : block ID; always b'##DZ'
    * ``reserved0`` - int : reserved bytes
    * ``block_len`` - int : block bytes size
    * ``links_nr`` - int : number of links
    * ``original_type`` - bytes : b'DT', b'SD', b'DI' or b'DV'
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
        "_transposed",
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

    def __init__(self, **kwargs) -> None:
        self._prevent_data_setitem = True
        self._transposed = False
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

            self._transposed = kwargs.get("transposed", False)
            self.data = data

        self._prevent_data_setitem = False
        self.return_unzipped = True

    def __setattr__(self, item: str, value: Any) -> None:
        if item == "data" and not self._prevent_data_setitem:
            data = value
            original_size = len(data)
            self.original_size = original_size

            if self.zip_type == v4c.FLAG_DZ_DEFLATE:
                data = compress(data, COMPRESSION_LEVEL)
            else:
                if not self._transposed:
                    cols = self.param
                    lines = original_size // cols

                    if lines * cols < original_size:
                        data = memoryview(data)
                        data = (
                            np.frombuffer(data[: lines * cols], dtype="B").reshape((lines, cols)).T.ravel().tobytes()
                        ) + data[lines * cols :]

                    else:
                        data = np.frombuffer(data, dtype=np.uint8).reshape((lines, cols)).T.ravel().tobytes()
                data = compress(data, COMPRESSION_LEVEL)

            zipped_size = len(data)
            self.zip_size = zipped_size
            self.block_len = zipped_size + v4c.DZ_COMMON_SIZE
            DataZippedBlock.__dict__[item].__set__(self, data)
            DataZippedBlock.__dict__["_transposed"].__set__(self, False)
        else:
            DataZippedBlock.__dict__[item].__set__(self, value)
            DataZippedBlock.__dict__["_transposed"].__set__(self, False)

    def __getattribute__(self, item: str) -> Any:
        if item == "data":
            if self.return_unzipped:
                data = DataZippedBlock.__dict__[item].__get__(self)
                original_size = self.original_size
                data = decompress(data, bufsize=original_size)
                if self.zip_type == v4c.FLAG_DZ_TRANPOSED_DEFLATE:
                    cols = self.param
                    lines = original_size // cols

                    if lines * cols < original_size:
                        data = memoryview(data)
                        data = (
                            np.frombuffer(data[: lines * cols], dtype=np.uint8)
                            .reshape((cols, lines))
                            .T.ravel()
                            .tobytes()
                        ) + data[lines * cols :]
                    else:
                        data = np.frombuffer(data, dtype=np.uint8).reshape((cols, lines)).T.ravel().tobytes()
            else:
                data = DataZippedBlock.__dict__[item].__get__(self)
            value = data
        else:
            value = DataZippedBlock.__dict__[item].__get__(self)
        return value

    def __setitem__(self, item: str, value: Any) -> None:
        self.__setattr__(item, value)

    def __getitem__(self, item: str) -> Any:
        return self.__getattribute__(item)

    def __str__(self) -> str:
        return f"""<DZBLOCK (address: {hex(self.address)}, original_size: {self.original_size}, zipped_size: {self.zip_size})>"""

    def __bytes__(self) -> bytes:
        self.return_unzipped = False
        data = (
            v4c.DZ_COMMON_p(
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
            )
            + self.data
        )
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
    * ``comment_addr`` - int : address of TXBLOCK/MDBLOCK that contains the
      data group comment
    * ``record_id_len`` - int : size of record ID used in case of unsorted
      data groups; can be 1, 2, 4 or 8
    * ``reserved1`` - int : reserved bytes

    Other attributes

    * ``address`` - int : data group address
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

    def __init__(self, **kwargs) -> None:
        self.comment = ""

        try:
            self.address = address = kwargs["address"]
            stream = kwargs["stream"]
            mapped = kwargs.get("mapped", False) or not is_file_like(stream)

            if mapped:
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

    def copy(self) -> DataGroup:
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

    def to_blocks(self, address: int, blocks: list[Any], defined_texts: dict[str, int]) -> int:
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

    def __getitem__(self, item: str) -> Any:
        return self.__getattribute__(item)

    def __setitem__(self, item: str, value: Any) -> None:
        self.__setattr__(item, value)

    def __bytes__(self) -> bytes:
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
    * ``data_block_nr`` - int : number of data blocks referenced by this list

    DLBLOCK specific fields

    * for equal length blocks

        * ``data_block_len`` - int : equal uncompressed size in bytes for all
          referenced data blocks; last block can be smaller

    * for variable length blocks

        * ``offset_<N>`` - int : byte offset of N-th data block

    Other attributes

    * ``address`` - int : data list address

    """

    def __init__(self, **kwargs) -> None:
        try:
            self.address = address = kwargs["address"]
            stream = kwargs["stream"]
            mapped = kwargs.get("mapped", False) or not is_file_like(stream)

            if mapped:
                (self.id, self.reserved0, self.block_len, self.links_nr) = COMMON_uf(stream, address)

                if self.id != b"##DL":
                    message = f'Expected "##DL" block @{hex(address)} but found "{self.id}"'

                    logger.exception(message)
                    raise MdfException(message)

                address += COMMON_SIZE

                links = unpack_from(f"<{self.links_nr}Q", stream, address)

                self.next_dl_addr = links[0]

                for i, addr in enumerate(links[1:]):
                    self[f"data_block_addr{i}"] = addr

                stream.seek(address + self.links_nr * 8)

                self.flags = stream.read(1)[0]
                if self.flags & v4c.FLAG_DL_EQUAL_LENGHT:
                    (self.reserved1, self.data_block_nr, self.data_block_len) = unpack("<3sIQ", stream.read(15))
                else:
                    (self.reserved1, self.data_block_nr) = unpack("<3sI", stream.read(7))
                    offsets = unpack(
                        f"<{self.links_nr - 1}Q",
                        stream.read((self.links_nr - 1) * 8),
                    )
                    for i, offset in enumerate(offsets):
                        self[f"offset_{i}"] = offset
            else:
                stream.seek(address)

                (self.id, self.reserved0, self.block_len, self.links_nr) = COMMON_u(stream.read(COMMON_SIZE))

                if self.id != b"##DL":
                    message = f'Expected "##DL" block @{hex(address)} but found "{self.id}"'

                    logger.exception(message)
                    raise MdfException(message)

                links = unpack(f"<{self.links_nr}Q", stream.read(self.links_nr * 8))

                self.next_dl_addr = links[0]

                for i, addr in enumerate(links[1:]):
                    self[f"data_block_addr{i}"] = addr

                self.flags = stream.read(1)[0]
                if self.flags & v4c.FLAG_DL_EQUAL_LENGHT:
                    (self.reserved1, self.data_block_nr, self.data_block_len) = unpack("<3sIQ", stream.read(15))
                else:
                    (self.reserved1, self.data_block_nr) = unpack("<3sI", stream.read(7))
                    offsets = unpack(
                        f"<{self.links_nr - 1}Q",
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
            self.data_block_nr = kwargs.get("data_block_nr", self.links_nr - 1)
            if self.flags & v4c.FLAG_DL_EQUAL_LENGHT:
                self.data_block_len = kwargs["data_block_len"]
            else:
                self.block_len = 40 + 8 * kwargs.get("links_nr", 2) - 8 + 8 * self.data_block_nr
                for i in range(self.data_block_nr):
                    self[f"offset_{i}"] = kwargs[f"offset_{i}"]

    def __getitem__(self, item: str) -> Any:
        return self.__getattribute__(item)

    def __setitem__(self, item: str, value: Any) -> None:
        self.__setattr__(item, value)

    def __bytes__(self) -> bytes:
        keys = ("id", "reserved0", "block_len", "links_nr", "next_dl_addr")
        keys += tuple(f"data_block_addr{i}" for i in range(self.links_nr - 1))
        if self.flags & v4c.FLAG_DL_EQUAL_LENGHT:
            keys += ("flags", "reserved1", "data_block_nr", "data_block_len")
            fmt = v4c.FMT_DATA_EQUAL_LIST.format(self.links_nr)
        else:
            keys += ("flags", "reserved1", "data_block_nr")
            keys += tuple(f"offset_{i}" for i in range(self.data_block_nr))
            fmt = v4c.FMT_DATA_LIST.format(self.links_nr, self.data_block_nr)

        result = pack(fmt, *[getattr(self, key) for key in keys])

        return result


class _EventBlockBase:
    __slots__ = (
        "address",
        "comment",
        "name",
        "parent",
        "group_name",
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
        "group_name_addr",
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
        "tool",
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

    def __init__(self, **kwargs) -> None:
        self.name = self.comment = self.group_name = ""
        self.scopes = []
        self.parent = None
        self.range_start = None

        if "stream" in kwargs:
            self.address = address = kwargs["address"]
            stream = kwargs["stream"]
            stream.seek(address)

            (self.id, self.reserved0, self.block_len, self.links_nr) = COMMON_u(stream.read(COMMON_SIZE))

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

            if self.flags & v4c.FLAG_EV_GROUP_NAME:
                self.group_name_addr = links[-1]

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
            self.sync_base = kwargs.get("sync_base", 1)
            self.sync_factor = kwargs.get("sync_factor", 1.0)

            if self.flags & v4c.FLAG_EV_GROUP_NAME:
                self.group_name_addr = kwargs.get("group_name_addr", 0)

        self.tool = extract_ev_tool(self.comment)

    def update_references(self, ch_map, cg_map):
        self.scopes.clear()
        for i in range(self.scope_nr):
            addr = self[f"scope_{i}_addr"]
            if addr in ch_map:
                self.scopes.append(ch_map[addr])
            elif addr in cg_map:
                self.scopes.append(cg_map[addr])
            else:
                message = "{} is not a valid CNBLOCK or CGBLOCK " "address for the event scope"
                message = message.format(hex(addr))
                logger.exception(message)
                raise MdfException(message)

    def __bytes__(self) -> bytes:
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

        if self.flags & v4c.FLAG_EV_GROUP_NAME:
            keys += ("group_name_addr",)

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

    def __getitem__(self, item: str) -> Any:
        return self.__getattribute__(item)

    def __setitem__(self, item: str, value: Any) -> None:
        self.__setattr__(item, value)

    def __str__(self) -> str:
        return f"EventBlock (name: {self.name}, comment: {self.comment}, address: {hex(self.address)}, scopes: {self.scopes}, fields: {super().__str__()})"

    @property
    def value(self):
        return self.sync_base * self.sync_factor

    @value.setter
    def value(self, val) -> None:
        self.sync_factor = val / self.sync_base

    def to_blocks(self, address: int, blocks: list[Any]) -> int:
        text = self.name
        if text:
            tx_block = TextBlock(text=text)
            self.name_addr = address
            tx_block.address = address
            address += tx_block.block_len
            blocks.append(tx_block)
        else:
            self.name_addr = 0

        text = self.comment
        if text:
            tx_block = TextBlock(text=text)
            self.comment_addr = address
            tx_block.address = address
            address += tx_block.block_len
            blocks.append(tx_block)
        else:
            self.comment_addr = 0

        if self.flags & v4c.FLAG_EV_GROUP_NAME:
            text = self.group_name
            if text:
                tx_block = TextBlock(text=text)
                self.group_name_addr = address
                tx_block.address = address
                address += tx_block.block_len
                blocks.append(tx_block)
            else:
                self.group_name_addr = 0

        blocks.append(self)
        self.address = address
        address += self.block_len

        return address


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

    def __init__(self, **kwargs) -> None:
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
            ) = unpack(v4c.FMT_IDENTIFICATION_BLOCK, stream.read(v4c.IDENTIFICATION_BLOCK_SIZE))

        except KeyError:
            version = kwargs.get("version", "4.00")
            self.file_identification = b"MDF     "
            self.version_str = f"{version}    ".encode()
            self.program_identification = "amdf{}".format(__version__.replace(".", "")).encode("utf-8")
            self.reserved0 = b"\0" * 4
            self.mdf_version = int(version.replace(".", ""))
            self.reserved1 = b"\0" * 30
            self.unfinalized_standard_flags = 0
            self.unfinalized_custom_flags = 0

    def __getitem__(self, item: str) -> Any:
        return self.__getattribute__(item)

    def __setitem__(self, item: str, value: Any) -> None:
        self.__setattr__(item, value)

    def __bytes__(self) -> bytes:
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

    def __init__(self, **kwargs) -> None:
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
            self.abs_time = kwargs.get("abs_time", int(time.time()) * 10**9)
            self.tz_offset = kwargs.get("tz_offset", 120)
            self.daylight_save_time = kwargs.get("daylight_save_time", 60)
            self.time_flags = kwargs.get("time_flags", 2)
            self.reserved1 = kwargs.get("reserved1", b"\x00" * 3)

            localtz = dateutil.tz.tzlocal()
            self.time_stamp = datetime.fromtimestamp(time.time(), tz=localtz)

    def to_blocks(self, address: int, blocks: list[Any], defined_texts: dict[str, int]) -> int:
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

    def __getitem__(self, item: str) -> Any:
        return self.__getattribute__(item)

    def __setitem__(self, item: str, value: Any) -> None:
        self.__setattr__(item, value)

    def __bytes__(self) -> bytes:
        result = pack(v4c.FMT_FILE_HISTORY, *[self[key] for key in v4c.KEYS_FILE_HISTORY])
        return result

    @property
    def time_stamp(self) -> datetime:
        """getter and setter the file history timestamp

        Returns
        -------
        timestamp : datetime.datetime
            start timestamp

        """

        timestamp = self.abs_time / 10**9
        if self.time_flags & v4c.FLAG_HD_LOCAL_TIME:
            tz = dateutil.tz.tzlocal()
        else:
            tz = timezone(timedelta(minutes=self.tz_offset + self.daylight_save_time))

        try:
            timestamp = datetime.fromtimestamp(timestamp, tz)

        except OverflowError:
            timestamp = datetime.fromtimestamp(0, tz) + timedelta(seconds=timestamp)

        return timestamp

    @time_stamp.setter
    def time_stamp(self, timestamp: datetime) -> None:
        if timestamp.tzinfo is None:
            self.time_flags = v4c.FLAG_HD_LOCAL_TIME
            self.abs_time = int(timestamp.timestamp() * 10**9)
            self.tz_offset = 0
            self.daylight_save_time = 0

        else:
            self.time_flags = v4c.FLAG_HD_TIME_OFFSET_VALID

            tzinfo = timestamp.tzinfo

            dst = tzinfo.dst(timestamp)
            if dst is not None:
                dst = int(tzinfo.dst(timestamp).total_seconds() / 60)
            else:
                dst = 0
            tz_offset = int(tzinfo.utcoffset(timestamp).total_seconds() / 60) - dst

            self.tz_offset = tz_offset
            self.daylight_save_time = dst
            self.abs_time = int(timestamp.timestamp() * 10**9)

    def __repr__(self):
        return f"FHBLOCK(time={self.time_stamp}, comment={self.comment})"


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

    def __init__(self, **kwargs) -> None:
        super().__init__()

        self._common_properties = {}
        self._other_elements = []
        self.description = ""

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
            self.abs_time = kwargs.get("abs_time", 0)
            self.tz_offset = kwargs.get("tz_offset", 0)
            self.daylight_save_time = kwargs.get("daylight_save_time", 0)
            self.time_flags = kwargs.get("time_flags", 0)
            self.time_quality = kwargs.get("time_quality", 0)
            self.flags = kwargs.get("flags", 0)
            self.reserved1 = kwargs.get("reserved4", 0)
            self.start_angle = kwargs.get("start_angle", 0)
            self.start_distance = kwargs.get("start_distance", 0)

            localtz = dateutil.tz.tzlocal()
            self.start_time = datetime.fromtimestamp(time.time(), tz=localtz)

    @property
    def comment(self):
        def common_properties_to_xml(root, common_properties):
            for name, value in common_properties.items():
                if isinstance(value, dict):
                    list_element = ET.SubElement(root, "tree", name=name)
                    common_properties_to_xml(list_element, value)

                else:
                    ET.SubElement(root, "e", name=name).text = value

        root = ET.Element("HDcomment")
        text = ET.SubElement(root, "TX")
        text.text = self.description
        common = ET.SubElement(root, "common_properties")

        common_properties_to_xml(common, self._common_properties)

        for elem in self._other_elements:
            root.append(elem)

        comment_xml = (
            ET.tostring(root, encoding="utf8", method="xml")
            .replace(b"<?xml version='1.0' encoding='utf8'?>\n", b"")
            .decode("utf-8")
        )

        comment_xml = minidom.parseString(comment_xml).toprettyxml(indent=" ")

        return "\n".join(line for line in comment_xml.splitlines()[1:] if line.strip())

    @comment.setter
    def comment(self, string):
        self._common_properties.clear()
        self._other_elements.clear()

        def parse_common_properties(root):
            root_name = root.get("name")
            info = {}
            if root.tag in ("list", "tree", "elist"):
                info[root_name] = {}
            try:
                for element in root:
                    name = element.get("name")

                    if element.tag == "e":
                        if root.tag == "tree":
                            info[root_name][name] = element.text or ""
                        else:
                            info[name] = element.text or ""

                    elif element.tag in ("list", "tree", "elist"):
                        info.update(parse_common_properties(element))

                    elif element.tag == "li":
                        info[root_name].update(parse_common_properties(element))

                    elif element.tag == "eli":
                        info[root_name][str(len(info[root_name]))] = element.text or ""
            except:
                print(format_exc())

            return info

        if string.startswith("<HDcomment"):
            comment = string
            try:
                comment_xml = ET.fromstring(re.sub(r" xmlns=[\'\"]http://www.asam.net/mdf/v4[\'\"]", "", comment))
            except ET.ParseError as e:
                self.description = string
                logger.error(f"could not parse header block comment; {e}")
            else:
                description = comment_xml.find(".//TX")
                if description is None:
                    self.description = ""
                else:
                    self.description = description.text or ""

                common_properties = comment_xml.find(".//common_properties")
                if common_properties is not None:
                    self._common_properties = parse_common_properties(common_properties)

                for element in comment_xml:
                    if element.tag not in ("TX", "common_properties"):
                        self._other_elements.append(element)

        else:
            self.description = string

    @property
    def author(self):
        return self._common_properties.get("author", "")

    @author.setter
    def author(self, value):
        self._common_properties["author"] = value

    @property
    def project(self):
        return self._common_properties.get("project", "")

    @project.setter
    def project(self, value):
        self._common_properties["project"] = value

    @property
    def department(self):
        return self._common_properties.get("department", "")

    @department.setter
    def department(self, value):
        self._common_properties["department"] = value

    @property
    def subject(self):
        return self._common_properties.get("subject", "")

    @subject.setter
    def subject(self, value):
        self._common_properties["subject"] = value

    def __getitem__(self, item: str) -> Any:
        return self.__getattribute__(item)

    def __setitem__(self, item: str, value: Any) -> None:
        self.__setattr__(item, value)

    @property
    def start_time(self) -> datetime:
        """getter and setter the measurement start timestamp

        Returns
        -------
        timestamp : datetime.datetime
            start timestamp

        """

        timestamp = self.abs_time / 10**9
        if self.time_flags & v4c.FLAG_HD_LOCAL_TIME:
            tz = dateutil.tz.tzlocal()
        else:
            tz = timezone(timedelta(minutes=self.tz_offset + self.daylight_save_time))

        try:
            timestamp = datetime.fromtimestamp(timestamp, tz)

        except OverflowError:
            timestamp = datetime.fromtimestamp(0, tz) + timedelta(seconds=timestamp)

        return timestamp

    @start_time.setter
    def start_time(self, timestamp: datetime) -> None:
        if timestamp.tzinfo is None:
            self.time_flags = v4c.FLAG_HD_LOCAL_TIME
            self.abs_time = int(timestamp.timestamp() * 10**9)
            self.tz_offset = 0
            self.daylight_save_time = 0

        else:
            self.time_flags = v4c.FLAG_HD_TIME_OFFSET_VALID

            tzinfo = timestamp.tzinfo

            dst = tzinfo.dst(timestamp)
            if dst is not None:
                dst = int(tzinfo.dst(timestamp).total_seconds() / 60)
            else:
                dst = 0
            tz_offset = int(tzinfo.utcoffset(timestamp).total_seconds() / 60) - dst

            self.tz_offset = tz_offset
            self.daylight_save_time = dst
            self.abs_time = int(timestamp.timestamp() * 10**9)

    def start_time_string(self):
        if self.time_flags & v4c.FLAG_HD_TIME_OFFSET_VALID:
            tz_offset = self.tz_offset / 60
            tz_offset_sign = "-" if tz_offset < 0 else "+"

            dst_offset = self.daylight_save_time / 60
            dst_offset_sign = "-" if dst_offset < 0 else "+"

            tz_information = f"[GMT{tz_offset_sign}{tz_offset:.2f} DST{dst_offset_sign}{dst_offset:.2f}h]"

            start_time = f'local time = {self.start_time.strftime("%d-%b-%Y %H:%M:%S + %fu")} {tz_information}'

        else:
            tzinfo = self.start_time.tzinfo

            dst = tzinfo.dst(self.start_time)
            if dst is not None:
                dst = int(tzinfo.dst(self.start_time).total_seconds() / 3600)
            else:
                dst = 0

            tz_offset = int(tzinfo.utcoffset(self.start_time).total_seconds() / 3600) - dst

            tz_offset_sign = "-" if tz_offset < 0 else "+"

            dst_offset = dst
            dst_offset_sign = "-" if dst_offset < 0 else "+"

            tz_information = f"[assumed GMT{tz_offset_sign}{tz_offset:.2f} DST{dst_offset_sign}{dst_offset:.2f}h]"

            start_time = f'local time = {self.start_time.strftime("%d-%b-%Y %H:%M:%S + %fu")} {tz_information}'

        return start_time

    def to_blocks(self, address: int, blocks: list[Any]) -> int:
        blocks.append(self)
        self.address = address
        address += self.block_len

        tx_block = TextBlock(text=self.comment, meta=True)
        self.comment_addr = address
        tx_block.address = address
        address += tx_block.block_len
        blocks.append(tx_block)

        return address

    def __bytes__(self) -> bytes:
        result = pack(v4c.FMT_HEADER_BLOCK, *[self[key] for key in v4c.KEYS_HEADER_BLOCK])
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

    def __init__(self, **kwargs) -> None:
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
            self.flags = kwargs.get("flags", 1)
            self.zip_type = kwargs.get("zip_type", 0)
            self.reserved1 = b"\x00" * 5

    def __getitem__(self, item: str) -> Any:
        return self.__getattribute__(item)

    def __setitem__(self, item: str, value: Any) -> None:
        self.__setattr__(item, value)

    def __bytes__(self) -> bytes:
        result = pack(v4c.FMT_HL_BLOCK, *[self[key] for key in v4c.KEYS_HL_BLOCK])
        return result


class _ListDataBase:
    __slots__ = (
        "address",
        "id",
        "reserved0",
        "block_len",
        "links_nr",
        "next_ld_addr",
        "flags",
        "data_block_nr",
        "data_block_len",
    )


class ListData(_ListDataBase):
    """
    *ListData* has the following attributes, that are also available as
    dict like key-value pairs

    LDBLOCK common fields

    * ``id`` - bytes : block ID; always b'##LD'
    * ``reserved0`` - int : reserved bytes
    * ``block_len`` - int : block bytes size
    * ``links_nr`` - int : number of links
    * ``next_ld_addr`` - int : address of next LDBLOCK
    * ``data_block_addr_<N>`` - int : address of N-th data block
      bits data block
    * ``flags`` - int : data list flags
    * ``data_block_nr`` - int : number of data blocks referenced by this list

    LDBLOCK specific fields

    * if invalidation data present flag is set

        * ``invalidation_bits_addr_<N>`` - int : address of N-th invalidation

    * for equal length blocks

        * ``data_block_len`` - int : equal uncompressed size in bytes for all
          referenced data blocks; last block can be smaller

    * for variable length blocks

        * ``offset_<N>`` - int : byte offset of N-th data block

    * if time values flag is set

        * ``time_value_<N>`` - int | float : first raw timestamp value of
          N-th data block

    * if angle values flag is set

        * ``angle_value_<N>`` - int | float : first raw angle value of
          N-th data block

    * if distance values flag is set

        * ``distance_value_<N>`` - int | float : first raw distance value of
          N-th data block

    Other attributes

    * ``address`` - int : data list address

    """

    def __init__(self, **kwargs) -> None:
        try:
            self.address = address = kwargs["address"]
            stream = kwargs["stream"]
            mapped = kwargs.get("mapped", False) or not is_file_like(stream)

            if mapped:
                (self.id, self.reserved0, self.block_len, self.links_nr) = COMMON_uf(stream, address)

                if self.id != b"##LD":
                    message = f'Expected "##LD" block @{hex(address)} but found "{self.id}"'

                    logger.exception(message)
                    raise MdfException(message)

                address += COMMON_SIZE

                links = unpack_from(f"<{self.links_nr}Q", stream, address)

                address += self.links_nr * 8

                self.flags, self.data_block_nr = unpack_from("<2I", stream, address)
                address += 8
                if self.flags & v4c.FLAG_LD_EQUAL_LENGHT:
                    (self.data_block_len,) = UINT64_uf(stream, address)
                    address += 8
                else:
                    offsets = unpack_from(f"<{self.data_block_nr}Q", stream, address)
                    address += self.data_block_nr * 8
                    for i, offset in enumerate(offsets):
                        self[f"offset_{i}"] = offset

                if self.flags & v4c.FLAG_LD_TIME_VALUES:
                    values = unpack_from(f"<{8 * self.data_block_nr}s", stream, address)
                    address += self.data_block_nr * 8
                    for i, value in enumerate(values):
                        self[f"time_value_{i}"] = value

                if self.flags & v4c.FLAG_LD_ANGLE_VALUES:
                    values = unpack_from(f"<{8 * self.data_block_nr}s", stream, address)
                    address += self.data_block_nr * 8
                    for i, value in enumerate(values):
                        self[f"angle_value_{i}"] = value

                if self.flags & v4c.FLAG_LD_DISTANCE_VALUES:
                    values = unpack_from(f"<{8 * self.data_block_nr}s", stream, address)
                    address += self.data_block_nr * 8
                    for i, value in enumerate(values):
                        self[f"distance_value_{i}"] = value

                self.next_ld_addr = links[0]

                for i in range(self.data_block_nr):
                    self[f"data_block_addr_{i}"] = links[i + 1]

                if self.flags & v4c.FLAG_LD_INVALIDATION_PRESENT:
                    for i in range(self.data_block_nr):
                        self[f"invalidation_bits_addr_{i}"] = links[self.data_block_nr + 1 + i]
            else:
                stream.seek(address)

                (self.id, self.reserved0, self.block_len, self.links_nr) = COMMON_u(stream.read(COMMON_SIZE))

                if self.id != b"##LD":
                    message = f'Expected "##LD" block @{hex(address)} but found "{self.id}"'

                    logger.exception(message)
                    raise MdfException(message)

                links = unpack(f"<{self.links_nr}Q", stream.read(self.links_nr * 8))

                self.flags, self.data_block_nr = unpack("<2I", stream(8))

                if self.flags & v4c.FLAG_LD_EQUAL_LENGHT:
                    (self.data_block_len,) = UINT64_u(stream.read(8))
                else:
                    offsets = unpack(f"<{self.data_block_nr}Q", stream.read(self.data_block_nr * 8))
                    for i, offset in enumerate(offsets):
                        self[f"offset_{i}"] = offset

                if self.flags & v4c.FLAG_LD_TIME_VALUES:
                    values = unpack(
                        f"<{8 * self.data_block_nr}s",
                        stream.read(self.data_block_nr * 8),
                    )
                    for i, value in enumerate(values):
                        self[f"time_value_{i}"] = value

                if self.flags & v4c.FLAG_LD_ANGLE_VALUES:
                    values = unpack(
                        f"<{8 * self.data_block_nr}s",
                        stream.read(self.data_block_nr * 8),
                    )
                    for i, value in enumerate(values):
                        self[f"angle_value_{i}"] = value

                if self.flags & v4c.FLAG_LD_DISTANCE_VALUES:
                    values = unpack(
                        f"<{8 * self.data_block_nr}s",
                        stream.read(self.data_block_nr * 8),
                    )
                    for i, value in enumerate(values):
                        self[f"distance_value_{i}"] = value

                self.next_ld_addr = links[0]

                for i in range(self.data_block_nr, 1):
                    self[f"data_block_addr_{i}"] = links[i]

                if self.flags & v4c.FLAG_LD_INVALIDATION_PRESENT:
                    for i in range(self.data_block_nr, self.data_block_nr + 1):
                        self[f"invalidation_bits_addr_{i}"] = links[i]

        except KeyError:
            self.address = 0

            self.id = b"##LD"
            self.reserved0 = 0

            self.data_block_nr = kwargs["data_block_nr"]
            self.flags = kwargs["flags"]
            self.data_block_len = kwargs["data_block_len"]
            self.next_ld_addr = 0

            for i in range(self.data_block_nr):
                self[f"data_block_addr_{i}"] = kwargs[f"data_block_addr_{i}"]
            if self.flags & v4c.FLAG_LD_INVALIDATION_PRESENT:
                self.links_nr = 2 * self.data_block_nr + 1

                for i in range(self.data_block_nr):
                    self[f"invalidation_bits_addr_{i}"] = kwargs[f"invalidation_bits_addr_{i}"]
            else:
                self.links_nr = self.data_block_nr + 1

            self.block_len = 24 + self.links_nr * 8 + 16

    def __getitem__(self, item: str) -> Any:
        return self.__getattribute__(item)

    def __setitem__(self, item: str, value: Any) -> None:
        self.__setattr__(item, value)

    def __bytes__(self) -> bytes:
        fmt = "<4sI3Q"
        keys = (
            "id",
            "reserved0",
            "block_len",
            "links_nr",
            "next_ld_addr",
        )

        fmt += f"{self.data_block_nr}Q"
        keys += tuple(f"data_block_addr_{i}" for i in range(self.data_block_nr))

        if self.flags & v4c.FLAG_LD_INVALIDATION_PRESENT:
            fmt += f"{self.data_block_nr}Q"
            keys += tuple(f"invalidation_bits_addr_{i}" for i in range(self.data_block_nr))

        fmt += "2I"
        keys += ("flags", "data_block_nr")

        if self.flags & v4c.FLAG_LD_EQUAL_LENGHT:
            fmt += "Q"
            keys += ("data_block_len",)

        else:
            fmt += f"<{self.data_block_nr}Q"
            keys += tuple(f"offset_{i}" for i in range(self.data_block_nr))

        if self.flags & v4c.FLAG_LD_TIME_VALUES:
            fmt += "8s" * self.data_block_nr
            keys += tuple(f"time_value_{i}" for i in range(self.data_block_nr))

        if self.flags & v4c.FLAG_LD_ANGLE_VALUES:
            fmt += "8s" * self.data_block_nr
            keys += tuple(f"angle_value_{i}" for i in range(self.data_block_nr))

        if self.flags & v4c.FLAG_LD_DISTANCE_VALUES:
            fmt += "8s" * self.data_block_nr
            keys += tuple(f"distance_value_{i}" for i in range(self.data_block_nr))

        result = pack(fmt, *[getattr(self, key) for key in keys])
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
    * ``comment_addr`` - int : address of TXBLOCK/MDBLOCK that contains the
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

    def __init__(self, **kwargs) -> None:
        self.name = self.path = self.comment = ""

        if "stream" in kwargs:
            stream = kwargs["stream"]
            mapped = kwargs["mapped"]
            try:
                block = kwargs["raw_bytes"]
                self.address = address = kwargs["address"]
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

            tx_map = kwargs["tx_map"]

            address = self.name_addr
            if address in tx_map:
                self.name = tx_map[address]
            else:
                self.name = get_text_v4(address=self.name_addr, stream=stream, mapped=mapped)
                tx_map[address] = self.name

            address = self.path_addr
            if address in tx_map:
                self.path = tx_map[address]
            else:
                self.path = get_text_v4(address=self.path_addr, stream=stream, mapped=mapped)
                tx_map[address] = self.path

            address = self.comment_addr
            if address in tx_map:
                self.comment = tx_map[address]
            else:
                self.comment = get_text_v4(address=self.comment_addr, stream=stream, mapped=mapped)
                tx_map[address] = self.comment

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

    def __getitem__(self, item: str) -> Any:
        return self.__getattribute__(item)

    def __setitem__(self, item: str, value: Any) -> None:
        self.__setattr__(item, value)

    def __contains__(self, item: str) -> bool:
        return hasattr(self, item)

    def copy(self) -> SourceInformation:
        source = SourceInformation(
            source_type=self.source_type,
            bus_type=self.bus_type,
        )
        source.name = self.name
        source.comment = self.comment
        source.path = self.path

        return source

    def metadata(self) -> str:
        max_len = max(len(key) for key in v4c.KEYS_SOURCE_INFORMATION)
        template = f"{{: <{max_len}}}: {{}}"

        metadata = []
        lines = f"""
name: {self.name}
path: {self.path}
address: {hex(self.address)}
comment: {self.comment}

""".split(
            "\n"
        )
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

            if key == "source_type":
                lines[-1] += f" = {v4c.SOURCE_TYPE_TO_STRING[self.source_type]}"
            elif key == "bus_type":
                lines[-1] += f" = {v4c.BUS_TYPE_TO_STRING[self.bus_type]}"

        for line in lines:
            if not line:
                metadata.append(line)
            else:
                for wrapped_line in wrap(line, width=120):
                    metadata.append(wrapped_line)

        return "\n".join(metadata)

    def to_blocks(
        self,
        address: int,
        blocks: list[Any],
        defined_texts: dict[str, int],
        si_map: dict[bytes, int],
    ) -> int:
        id_ = id(self)
        if id_ in si_map:
            return address

        text = self.name
        if text:
            if text in defined_texts:
                self.name_addr = defined_texts[text]
            else:
                tx_block = TextBlock(
                    text=text.encode("utf-8", "replace"),
                    meta=False,
                    safe=True,
                )
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
                tx_block = TextBlock(
                    text=text.encode("utf-8", "replace"),
                    meta=False,
                    safe=True,
                )
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
                meta = text.startswith("<SI")
                tx_block = TextBlock(
                    text=text.encode("utf-8", "replace"),
                    meta=meta,
                    safe=True,
                )
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
            si_map[bts] = si_map[id(self)] = address
            self.address = address
            address += self.block_len

        return address

    @classmethod
    def from_common_source(cls, source: Source) -> SourceInformation:
        obj = cls()
        obj.name = source.name
        obj.path = source.path
        obj.comment = source.comment
        obj.source_type = source.source_type
        obj.bus_type = source.bus_type

        return obj

    def __bytes__(self) -> bytes:
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

    def __str__(self) -> str:
        return f"<SourceInformation (name: {self.name}, path: {self.path}, comment: {self.comment}, address: {hex(self.address)}, fields: {block_fields(self)})>"


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

    def __init__(self, **kwargs) -> None:
        if "safe" in kwargs:
            self.address = 0
            text = kwargs["text"]
            size = len(text)
            self.id = b"##MD" if kwargs["meta"] else b"##TX"
            self.reserved0 = 0
            self.links_nr = 0
            self.text = text

            self.block_len = size + 32 - size % 8

        elif "stream" in kwargs:
            stream = kwargs["stream"]
            mapped = kwargs.get("mapped", False) or not is_file_like(stream)
            self.address = address = kwargs["address"]

            if mapped:
                (self.id, self.reserved0, self.block_len, self.links_nr) = COMMON_uf(stream, address)

                size = self.block_len - COMMON_SIZE

                if self.id not in (b"##TX", b"##MD"):
                    message = f'Expected "##TX" or "##MD" block @{hex(address)} but found "{self.id}"'
                    logger.exception(message)
                    raise MdfException(message)

                self.text = text = stream[address + COMMON_SIZE : address + self.block_len]

            else:
                stream.seek(address)
                (self.id, self.reserved0, self.block_len, self.links_nr) = COMMON_u(stream.read(COMMON_SIZE))

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

    def __getitem__(self, item: str) -> Any:
        return self.__getattribute__(item)

    def __setitem__(self, item: str, value: Any) -> None:
        self.__setattr__(item, value)

    def __bytes__(self) -> bytes:
        return pack(
            f"<4sI2Q{self.block_len - COMMON_SIZE}s",
            self.id,
            self.reserved0,
            self.block_len,
            self.links_nr,
            self.text,
        )

    def __repr__(self) -> str:
        return (
            f"TextBlock(id={self.id},"
            f"reserved0={self.reserved0}, "
            f"block_len={self.block_len}, "
            f"links_nr={self.links_nr} "
            f"text={self.text})"
        )

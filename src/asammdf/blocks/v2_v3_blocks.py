"""Classes that implement the blocks for MDF versions 2 and 3"""

from collections.abc import Iterator, Sequence
from datetime import datetime, timedelta, timezone
from getpass import getuser
import logging
from struct import pack, unpack, unpack_from
import sys
from textwrap import wrap
from traceback import format_exc
import typing
from typing import Final
import xml.etree.ElementTree as ET

import dateutil.tz
from numexpr import evaluate
import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing_extensions import Any, overload, SupportsBytes, TypedDict, Unpack

from .. import tool
from . import utils
from . import v2_v3_constants as v23c
from .utils import (
    BlockKwargs,
    get_fields,
    get_text_v3,
    MdfException,
    UINT16_u,
    UINT16_uf,
)

try:
    from sympy import lambdify, symbols

except:
    lambdify, symbols = None, None

SEEK_START: Final = v23c.SEEK_START
SEEK_END: Final = v23c.SEEK_END


CHANNEL_DISPLAYNAME_u = v23c.CHANNEL_DISPLAYNAME_u
CHANNEL_DISPLAYNAME_uf = v23c.CHANNEL_DISPLAYNAME_uf
CHANNEL_LONGNAME_u = v23c.CHANNEL_LONGNAME_u
CHANNEL_LONGNAME_uf = v23c.CHANNEL_LONGNAME_uf
CHANNEL_SHORT_u = v23c.CHANNEL_SHORT_u
CHANNEL_SHORT_uf = v23c.CHANNEL_SHORT_uf
COMMON_uf = v23c.COMMON_uf
COMMON_u = v23c.COMMON_u
CONVERSION_COMMON_SHORT_uf = v23c.CONVERSION_COMMON_SHORT_uf
SOURCE_COMMON_uf = v23c.SOURCE_COMMON_uf
SOURCE_EXTRA_ECU_uf = v23c.SOURCE_EXTRA_ECU_uf
SOURCE_EXTRA_VECTOR_uf = v23c.SOURCE_EXTRA_VECTOR_uf
SOURCE_COMMON_u = v23c.SOURCE_COMMON_u
SOURCE_EXTRA_ECU_u = v23c.SOURCE_EXTRA_ECU_u
SOURCE_EXTRA_VECTOR_u = v23c.SOURCE_EXTRA_VECTOR_u


logger = logging.getLogger("asammdf")


__all__ = [
    "Channel",
    "ChannelConversion",
    "ChannelDependency",
    "ChannelExtension",
    "ChannelGroup",
    "DataBlock",
    "DataGroup",
    "FileIdentificationBlock",
    "HeaderBlock",
    "ProgramBlock",
    "TextBlock",
    "TriggerBlock",
]


class ChannelKwargs(BlockKwargs, total=False):
    parsed_strings: tuple[str, dict[str, str]] | None
    cc_map: dict[bytes | int, "ChannelConversion"]
    si_map: dict[bytes | int, "ChannelExtension"]
    block_len: int
    next_ch_addr: int
    conversion_addr: int
    source_addr: int
    component_addr: int
    comment_addr: int
    channel_type: int
    short_name: bytes
    description: bytes
    start_offset: int
    bit_count: int
    data_type: int
    range_flag: int
    min_raw_value: float
    max_raw_value: float
    sampling_rate: float
    long_name_addr: int
    display_name_addr: int
    additional_byte_offset: int


class Channel:
    """CNBLOCK class.

    If the `load_metadata` keyword argument is not provided or is False,
    then the conversion, source and display name information is not processed.

    CNBLOCK fields:

    * ``id`` - bytes : block ID; always b'CN'
    * ``block_len`` - int : block bytes size
    * ``next_ch_addr`` - int : next CNBLOCK address
    * ``conversion_addr`` - int : address of channel conversion block
    * ``source_addr`` - int : address of channel source block
    * ``component_addr`` - int : address of dependency block (CDBLOCK) of this
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
    * ``additional_byte_offset`` - int : additional Byte offset of the channel
      in the data record

    Other attributes:

    * ``address`` - int : block address inside MDF file
    * ``comment`` - str : channel comment
    * ``conversion`` - ChannelConversion : channel conversion; None if the
      channel has no conversion
    * ``display_names`` - dict : channel display names

        .. versionchanged:: 7.0.0

            Changed from str to dict.

    * ``name`` - str : full channel name
    * ``source`` - SourceInformation : channel source information; None if the
      channel has no source information

    Parameters
    ----------
    address : int
        Block address; to be used for objects created from file.
    stream : handle
        File handle; to be used for objects created from file.
    load_metadata : bool
        Option to load conversion, source and display_names; default is True.
    for dynamically created objects :
        See the key-value pairs.

    Examples
    --------
    >>> with open('test.mdf', 'rb') as mdf:
    ...     ch1 = Channel(stream=mdf, address=0xBA52)
    >>> ch2 = Channel()
    >>> ch1.name
    'VehicleSpeed'
    >>> ch1['id']
    b'CN'
    """

    __slots__ = (
        "additional_byte_offset",
        "address",
        "bit_count",
        "block_len",
        "channel_type",
        "comment",
        "comment_addr",
        "component_addr",
        "conversion",
        "conversion_addr",
        "data_type",
        "description",
        "display_name_addr",
        "display_names",
        "dtype_fmt",
        "id",
        "long_name_addr",
        "max_raw_value",
        "min_raw_value",
        "name",
        "next_ch_addr",
        "range_flag",
        "sampling_rate",
        "short_name",
        "source",
        "source_addr",
        "start_offset",
        "unit",
    )

    def __init__(self, **kwargs: Unpack[ChannelKwargs]) -> None:
        super().__init__()

        self.name = self.comment = self.unit = ""
        self.display_names: dict[str, str] = {}
        self.conversion = self.source = None
        self.dtype_fmt: np.dtype[Any] | None = None

        try:
            stream = kwargs["stream"]
            self.address = address = kwargs["address"]
            mapped = kwargs.get("mapped", False)

            if utils.stream_is_mmap(stream, mapped):
                (size,) = UINT16_uf(stream, address + 2)

                if size == v23c.CN_DISPLAYNAME_BLOCK_SIZE:
                    (
                        self.id,
                        self.block_len,
                        self.next_ch_addr,
                        self.conversion_addr,
                        self.source_addr,
                        self.component_addr,
                        self.comment_addr,
                        self.channel_type,
                        self.short_name,
                        self.description,
                        self.start_offset,
                        self.bit_count,
                        self.data_type,
                        self.range_flag,
                        self.min_raw_value,
                        self.max_raw_value,
                        self.sampling_rate,
                        self.long_name_addr,
                        self.display_name_addr,
                        self.additional_byte_offset,
                    ) = CHANNEL_DISPLAYNAME_uf(stream, address)

                    parsed_strings = kwargs["parsed_strings"]

                    if parsed_strings is None:
                        addr = self.long_name_addr
                        if addr:
                            self.name = get_text_v3(address=addr, stream=stream, mapped=mapped)
                        else:
                            self.name = self.short_name.decode("latin-1").strip(" \t\n\r\0")

                        addr = self.display_name_addr
                        if addr:
                            self.display_names = {
                                get_text_v3(address=addr, stream=stream, mapped=mapped): "display_name",
                            }

                    else:
                        self.name, self.display_names = parsed_strings

                elif size == v23c.CN_LONGNAME_BLOCK_SIZE:
                    (
                        self.id,
                        self.block_len,
                        self.next_ch_addr,
                        self.conversion_addr,
                        self.source_addr,
                        self.component_addr,
                        self.comment_addr,
                        self.channel_type,
                        self.short_name,
                        self.description,
                        self.start_offset,
                        self.bit_count,
                        self.data_type,
                        self.range_flag,
                        self.min_raw_value,
                        self.max_raw_value,
                        self.sampling_rate,
                        self.long_name_addr,
                    ) = CHANNEL_LONGNAME_uf(stream, address)

                    parsed_strings = kwargs["parsed_strings"]

                    if parsed_strings is None:
                        addr = self.long_name_addr
                        if addr:
                            self.name = get_text_v3(address=addr, stream=stream, mapped=mapped)
                        else:
                            self.name = self.short_name.decode("latin-1").strip(" \t\n\r\0")

                    else:
                        self.name, self.display_names = parsed_strings

                else:
                    (
                        self.id,
                        self.block_len,
                        self.next_ch_addr,
                        self.conversion_addr,
                        self.source_addr,
                        self.component_addr,
                        self.comment_addr,
                        self.channel_type,
                        self.short_name,
                        self.description,
                        self.start_offset,
                        self.bit_count,
                        self.data_type,
                        self.range_flag,
                        self.min_raw_value,
                        self.max_raw_value,
                        self.sampling_rate,
                    ) = CHANNEL_SHORT_uf(stream, address)

                    self.name = self.short_name.decode("latin-1").strip(" \t\n\r\0")

                cc_map = kwargs.get("cc_map", {})
                si_map = kwargs.get("si_map", {})

                address = self.conversion_addr
                if address:
                    try:
                        if address in cc_map:
                            conv = cc_map[address]
                        else:
                            (size,) = UINT16_uf(stream, address + 2)
                            raw_bytes = stream[address : address + size]

                            if raw_bytes in cc_map:
                                conv = cc_map[raw_bytes]
                            else:
                                conv = ChannelConversion(
                                    raw_bytes=raw_bytes,
                                    stream=stream,
                                    address=address,
                                    mapped=mapped,
                                )
                                cc_map[raw_bytes] = cc_map[address] = conv
                    except:
                        logger.warning(
                            f"Channel conversion parsing error: {format_exc()}. The error is ignored and the channel conversion is None"
                        )
                        conv = None
                    self.conversion = conv

                address = self.source_addr
                if address:
                    try:
                        if address in si_map:
                            source = si_map[address]
                        else:
                            raw_bytes = stream[address : address + v23c.CE_BLOCK_SIZE]

                            if raw_bytes in si_map:
                                source = si_map[raw_bytes]
                            else:
                                source = ChannelExtension(
                                    raw_bytes=raw_bytes,
                                    stream=stream,
                                    address=address,
                                    mapped=mapped,
                                )
                                si_map[raw_bytes] = si_map[address] = source
                    except:
                        logger.warning(
                            f"Channel source parsing error: {format_exc()}. The error is ignored and the channel source is None"
                        )
                        source = None

                    self.source = source

                self.comment = get_text_v3(address=self.comment_addr, stream=stream, mapped=mapped)
            else:
                stream.seek(address + 2)
                (size,) = UINT16_u(stream.read(2))
                stream.seek(address)
                block = stream.read(size)

                if size == v23c.CN_DISPLAYNAME_BLOCK_SIZE:
                    (
                        self.id,
                        self.block_len,
                        self.next_ch_addr,
                        self.conversion_addr,
                        self.source_addr,
                        self.component_addr,
                        self.comment_addr,
                        self.channel_type,
                        self.short_name,
                        self.description,
                        self.start_offset,
                        self.bit_count,
                        self.data_type,
                        self.range_flag,
                        self.min_raw_value,
                        self.max_raw_value,
                        self.sampling_rate,
                        self.long_name_addr,
                        self.display_name_addr,
                        self.additional_byte_offset,
                    ) = CHANNEL_DISPLAYNAME_u(block)

                    parsed_strings = kwargs["parsed_strings"]

                    if parsed_strings is None:
                        addr = self.long_name_addr
                        if addr:
                            self.name = get_text_v3(address=addr, stream=stream, mapped=mapped)
                        else:
                            self.name = self.short_name.decode("latin-1").strip(" \t\n\r\0")

                        addr = self.display_name_addr
                        if addr:
                            self.display_names = {
                                get_text_v3(address=addr, stream=stream, mapped=mapped): "display_name",
                            }

                    else:
                        self.name, self.display_names = parsed_strings

                elif size == v23c.CN_LONGNAME_BLOCK_SIZE:
                    (
                        self.id,
                        self.block_len,
                        self.next_ch_addr,
                        self.conversion_addr,
                        self.source_addr,
                        self.component_addr,
                        self.comment_addr,
                        self.channel_type,
                        self.short_name,
                        self.description,
                        self.start_offset,
                        self.bit_count,
                        self.data_type,
                        self.range_flag,
                        self.min_raw_value,
                        self.max_raw_value,
                        self.sampling_rate,
                        self.long_name_addr,
                    ) = CHANNEL_LONGNAME_u(block)

                    parsed_strings = kwargs["parsed_strings"]

                    if parsed_strings is None:
                        addr = self.long_name_addr
                        if addr:
                            self.name = get_text_v3(address=addr, stream=stream, mapped=mapped)
                        else:
                            self.name = self.short_name.decode("latin-1").strip(" \t\n\r\0")

                    else:
                        self.name, self.display_names = parsed_strings

                else:
                    (
                        self.id,
                        self.block_len,
                        self.next_ch_addr,
                        self.conversion_addr,
                        self.source_addr,
                        self.component_addr,
                        self.comment_addr,
                        self.channel_type,
                        self.short_name,
                        self.description,
                        self.start_offset,
                        self.bit_count,
                        self.data_type,
                        self.range_flag,
                        self.min_raw_value,
                        self.max_raw_value,
                        self.sampling_rate,
                    ) = CHANNEL_SHORT_u(block)

                    self.name = self.short_name.decode("latin-1").strip(" \t\n\r\0")

                cc_map = kwargs.get("cc_map", {})
                si_map = kwargs.get("si_map", {})

                address = self.conversion_addr
                if address:
                    try:
                        if address in cc_map:
                            conv = cc_map[address]
                        else:
                            stream.seek(address + 2)
                            (size,) = UINT16_u(stream.read(2))
                            stream.seek(address)
                            raw_bytes = stream.read(size)

                            if raw_bytes in cc_map:
                                conv = cc_map[raw_bytes]
                            else:
                                conv = ChannelConversion(
                                    raw_bytes=raw_bytes,
                                    stream=stream,
                                    address=address,
                                    mapped=mapped,
                                )
                                cc_map[raw_bytes] = cc_map[address] = conv
                    except:
                        logger.warning(
                            f"Channel conversion parsing error: {format_exc()}. The error is ignored and the channel conversion is None"
                        )
                        conv = None

                    self.conversion = conv

                address = self.source_addr
                if address:
                    try:
                        if address in si_map:
                            source = si_map[address]
                        else:
                            stream.seek(address)
                            raw_bytes = stream.read(v23c.CE_BLOCK_SIZE)

                            if raw_bytes in si_map:
                                source = si_map[raw_bytes]
                            else:
                                source = ChannelExtension(
                                    raw_bytes=raw_bytes,
                                    stream=stream,
                                    address=address,
                                    mapped=mapped,
                                )
                                si_map[raw_bytes] = si_map[address] = source
                    except:
                        logger.warning(
                            f"Channel source parsing error: {format_exc()}. The error is ignored and the channel source is None"
                        )
                        source = None
                    self.source = source

                self.comment = get_text_v3(address=self.comment_addr, stream=stream, mapped=mapped)

            if self.id != b"CN":
                message = f'Expected "CN" block @{hex(address)} but found "{self.id!r}"'
                logger.exception(message)
                raise MdfException(message)

        except KeyError:
            self.address = 0
            self.id = b"CN"
            self.block_len = kwargs.get("block_len", v23c.CN_DISPLAYNAME_BLOCK_SIZE)
            self.next_ch_addr = kwargs.get("next_ch_addr", 0)
            self.conversion_addr = kwargs.get("conversion_addr", 0)
            self.source_addr = kwargs.get("source_addr", 0)
            self.component_addr = kwargs.get("component_addr", 0)
            self.comment_addr = kwargs.get("comment_addr", 0)
            self.channel_type = kwargs.get("channel_type", 0)
            self.short_name = kwargs.get("short_name", (b"\0" * 32))
            self.description = kwargs.get("description", (b"\0" * 128))
            self.start_offset = kwargs.get("start_offset", 0)
            self.bit_count = kwargs.get("bit_count", 8)
            self.data_type = kwargs.get("data_type", 0)
            self.range_flag = kwargs.get("range_flag", 1)
            self.min_raw_value = kwargs.get("min_raw_value", 0)
            self.max_raw_value = kwargs.get("max_raw_value", 0)
            self.sampling_rate = kwargs.get("sampling_rate", 0)
            if self.block_len >= v23c.CN_LONGNAME_BLOCK_SIZE:
                self.long_name_addr = kwargs.get("long_name_addr", 0)
            if self.block_len >= v23c.CN_DISPLAYNAME_BLOCK_SIZE:
                self.display_name_addr = kwargs.get("display_name_addr", 0)
                self.additional_byte_offset = kwargs.get("additional_byte_offset", 0)

        if self.name in self.display_names:
            del self.display_names[self.name]

    def to_blocks(
        self,
        address: int,
        blocks: list[bytes | SupportsBytes],
        defined_texts: dict[bytes | str, int],
        cc_map: dict[bytes, int],
        si_map: dict[bytes, int],
    ) -> int:
        text = self.name
        if self.block_len >= v23c.CN_LONGNAME_BLOCK_SIZE:
            if len(text) > 31:
                if text in defined_texts:
                    self.long_name_addr = defined_texts[text]
                else:
                    tx_block = TextBlock(text=text)
                    self.long_name_addr = address
                    defined_texts[text] = address
                    tx_block.address = address
                    address += tx_block.block_len
                    blocks.append(tx_block)
            else:
                self.long_name_addr = 0

        self.short_name = text.encode("latin-1", "backslashreplace")[:31]

        text = list(self.display_names)[0] if self.display_names else ""
        if self.block_len >= v23c.CN_DISPLAYNAME_BLOCK_SIZE:
            if text:
                if text in defined_texts:
                    self.display_name_addr = defined_texts[text]
                else:
                    tx_block = TextBlock(text=text)
                    self.display_name_addr = address
                    defined_texts[text] = address
                    tx_block.address = address
                    address += tx_block.block_len
                    blocks.append(tx_block)
            else:
                self.display_name_addr = 0

        text = self.comment
        if text:
            if len(text) < 128:
                self.description = text.encode("latin-1", "backslashreplace")[:127]
                self.comment_addr = 0
            else:
                if text in defined_texts:
                    self.comment_addr = defined_texts[text]
                else:
                    tx_block = TextBlock(text=text)
                    self.comment_addr = address
                    defined_texts[text] = address
                    tx_block.address = address
                    address += tx_block.block_len
                    blocks.append(tx_block)
                self.description = b"\0"
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

    def metadata(self) -> str:
        max_len = max(len(key) for key in self)
        template = f"{{: <{max_len}}}: {{}}"

        metadata: list[str] = []
        lines = f"""
name: {self.name}
display names: {self.display_names}
address: {hex(self.address)}
comment: {self.comment}

""".split("\n")

        keys = (
            "id",
            "block_len",
            "next_ch_addr",
            "conversion_addr",
            "source_addr",
            "component_addr",
            "comment_addr",
            "channel_type",
            "short_name",
            "description",
            "start_offset",
            "bit_count",
            "data_type",
            "range_flag",
            "min_raw_value",
            "max_raw_value",
            "sampling_rate",
            "long_name_addr",
            "display_name_addr",
            "additional_byte_offset",
        )

        for key in keys:
            if not hasattr(self, key):
                continue
            val = getattr(self, key)
            if key.endswith("addr") or key.startswith("text_"):
                lines.append(template.format(key, hex(val)))
            elif isinstance(val, float):
                lines.append(template.format(key, val))
            else:
                if isinstance(val, bytes):
                    lines.append(template.format(key, val.strip(b"\0")))
                else:
                    lines.append(template.format(key, val))

            if key == "data_type":
                lines[-1] += f" = {v23c.DATA_TYPE_TO_STRING[self.data_type]}"
            elif key == "channel_type":
                lines[-1] += f" = {v23c.CHANNEL_TYPE_TO_STRING[self.channel_type]}"

        for line in lines:
            if not line:
                metadata.append(line)
            else:
                for wrapped_line in wrap(line, width=120):
                    metadata.append(wrapped_line)

        return "\n".join(metadata)

    def __bytes__(self) -> bytes:
        block_len = self.block_len
        if block_len == v23c.CN_DISPLAYNAME_BLOCK_SIZE:
            return v23c.CHANNEL_DISPLAYNAME_p(
                self.id,
                self.block_len,
                self.next_ch_addr,
                self.conversion_addr,
                self.source_addr,
                self.component_addr,
                self.comment_addr,
                self.channel_type,
                self.short_name,
                self.description,
                self.start_offset,
                self.bit_count,
                self.data_type,
                self.range_flag,
                self.min_raw_value,
                self.max_raw_value,
                self.sampling_rate,
                self.long_name_addr,
                self.display_name_addr,
                self.additional_byte_offset,
            )
        elif block_len == v23c.CN_LONGNAME_BLOCK_SIZE:
            return v23c.CHANNEL_LONGNAME_p(
                self.id,
                self.block_len,
                self.next_ch_addr,
                self.conversion_addr,
                self.source_addr,
                self.component_addr,
                self.comment_addr,
                self.channel_type,
                self.short_name,
                self.description,
                self.start_offset,
                self.bit_count,
                self.data_type,
                self.range_flag,
                self.min_raw_value,
                self.max_raw_value,
                self.sampling_rate,
                self.long_name_addr,
            )
        else:
            return v23c.CHANNEL_SHORT_p(
                self.id,
                self.block_len,
                self.next_ch_addr,
                self.conversion_addr,
                self.source_addr,
                self.component_addr,
                self.comment_addr,
                self.channel_type,
                self.short_name,
                self.description,
                self.start_offset,
                self.bit_count,
                self.data_type,
                self.range_flag,
                self.min_raw_value,
                self.max_raw_value,
                self.sampling_rate,
            )

    def __getitem__(self, item: str) -> object:
        return self.__getattribute__(item)

    def __setitem__(self, item: str, value: object) -> None:
        self.__setattr__(item, value)

    def __contains__(self, item: str) -> bool:
        return hasattr(self, item)

    def __iter__(self) -> Iterator[str]:
        for attr in dir(self):
            if attr[:2] + attr[-2:] == "____":
                continue
            try:
                if callable(getattr(self, attr)):
                    continue
                yield attr
            except AttributeError:
                continue

    def __lt__(self, other: "Channel") -> int:
        self_start = self.start_offset
        other_start = other.start_offset
        try:
            self_additional_offset = self.additional_byte_offset
            if self_additional_offset:
                self_start += 8 * self_additional_offset
            other_additional_offset = other.additional_byte_offset
            if other_additional_offset:
                other_start += 8 * other_additional_offset
        except AttributeError:
            pass

        if self_start < other_start:
            result = 1
        elif self_start == other_start:
            if self.bit_count >= other.bit_count:
                result = 1
            else:
                result = 0
        else:
            result = 0
        return result

    def __repr__(self) -> str:
        fields = []
        for attr in dir(self):
            if attr[:2] + attr[-2:] == "____":
                continue
            try:
                if callable(getattr(self, attr)):
                    continue
                fields.append(f"{attr}:{getattr(self, attr)}")
            except AttributeError:
                continue
        return f"Channel (name: {self.name}, display names: {self.display_names}, comment: {self.comment}, address: {hex(self.address)}, fields: {fields})"


class _ChannelConversionBase:
    __slots__ = (
        "P1",
        "P2",
        "P3",
        "P4",
        "P5",
        "P6",
        "P7",
        "a",
        "address",
        "b",
        "block_len",
        "comment_addr",
        "conversion_type",
        "flags",
        "formula",
        "formula_field",
        "id",
        "inv_conv_addr",
        "max_phy_value",
        "min_phy_value",
        "precision",
        "ref_param_nr",
        "referenced_blocks",
        "reserved0",
        "unit",
        "unit_field",
        "val_param_nr",
    )


class ChannelConversionKwargs(BlockKwargs, total=False):
    raw_bytes: bytes
    unit: bytes
    conversion_type: int
    range_flag: int
    min_phy_value: float
    max_phy_value: float
    b: float
    a: float
    P1: float
    P2: float
    P3: float
    P4: float
    P5: float
    P6: float
    P7: float
    formula: bytes | str
    ref_param_nr: int
    CANapeHiddenExtra: bytes
    default_addr: bytes


class ChannelConversion(_ChannelConversionBase):
    """CCBLOCK class.

    `ChannelConversion` has the following common fields:

    * ``id`` - bytes : block ID; always b'CC'
    * ``block_len`` - int : block bytes size
    * ``range_flag`` - int : value range valid flag
    * ``min_phy_value`` - float : min raw value of all samples
    * ``max_phy_value`` - float : max raw value of all samples
    * ``unit`` - bytes : physical unit
    * ``conversion_type`` - int : integer code for conversion type
    * ``ref_param_nr`` - int : number of referenced parameters

    `ChannelConversion` has the following specific fields:

    * linear conversion

      * ``a`` - float : factor
      * ``b`` - float : offset
      * ``CANapeHiddenExtra`` - bytes : sometimes CANape appends extra
        information; not compliant with MDF specs

    * algebraic conversion

      * ``formula`` - bytes : equation as string

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

    Other attributes:

    * ``address`` - int : block address inside MDF file
    * ``formula`` - str : formula string in case of algebraic conversion
    * ``referenced_blocks`` - list : list of CCBLOCK/TXBLOCK referenced by the
      conversion
    * ``unit`` - str : physical unit

    Parameters
    ----------
    address : int
        Block address inside MDF file.
    raw_bytes : bytes
        Complete block read from disk.
    stream : file handle
        MDF file handle.
    for dynamically created objects :
        See the key-value pairs.

    Examples
    --------
    >>> with open('test.mdf', 'rb') as mdf:
    ...     cc1 = ChannelConversion(stream=mdf, address=0xBA52)
    >>> cc2 = ChannelConversion(conversion_type=0)
    >>> cc1['b'], cc1['a']
    0, 100.0
    """

    def __init__(self, **kwargs: Unpack[ChannelConversionKwargs]) -> None:
        super().__init__()

        self.is_user_defined = False

        self.unit = self.formula = ""

        self.referenced_blocks: dict[str, bytes | ChannelConversion] = {}

        if "raw_bytes" in kwargs or "stream" in kwargs:
            mapped = kwargs.get("mapped", False)
            self.address = address = kwargs.get("address", 0)
            try:
                block = kwargs["raw_bytes"]
                (self.id, self.block_len) = COMMON_uf(block)
                size = self.block_len
                block_size = len(block)
                block = block[4:]
                stream = kwargs["stream"]

            except KeyError:
                stream = kwargs["stream"]
                if utils.stream_is_mmap(stream, mapped):
                    (self.id, self.block_len) = COMMON_uf(stream, address)
                    block_size = size = self.block_len
                    block = stream[address + 4 : address + size]
                else:
                    stream.seek(address)
                    block = stream.read(4)
                    (self.id, self.block_len) = COMMON_u(block)

                    block_size = size = self.block_len
                    block = stream.read(size - 4)

            (
                self.range_flag,
                self.min_phy_value,
                self.max_phy_value,
                self.unit_field,
                self.conversion_type,
                self.ref_param_nr,
            ) = CONVERSION_COMMON_SHORT_uf(block)

            self.unit = self.unit_field.decode("latin-1").strip(" \t\r\n\0")

            conv_type = self.conversion_type

            if conv_type == v23c.CONVERSION_TYPE_LINEAR:
                (self.b, self.a) = typing.cast(
                    tuple[float, float], unpack_from("<2d", block, v23c.CC_COMMON_SHORT_SIZE)
                )
                if size != v23c.CC_LIN_BLOCK_SIZE:
                    self.CANapeHiddenExtra = block[v23c.CC_LIN_BLOCK_SIZE - 4 :]
                    size = len(self.CANapeHiddenExtra)
                    nr = size // 40
                    values = unpack_from("<" + "d32s" * nr, block, v23c.CC_COMMON_SHORT_SIZE)

                    for i in range(nr):
                        (self[f"param_val_{i}"], self[f"text_{i}"]) = (
                            values[i * 2],
                            values[2 * i + 1],
                        )

            elif conv_type == v23c.CONVERSION_TYPE_NONE:
                pass

            elif conv_type == v23c.CONVERSION_TYPE_FORMULA:
                self.formula_field = block[v23c.CC_COMMON_SHORT_SIZE :]
                self.formula = self.formula_field.decode("latin-1").strip(" \t\r\n\0").replace("x", "X")
                if "X1" not in self.formula:
                    self.formula = self.formula.replace("X", "X1")

            elif conv_type in (v23c.CONVERSION_TYPE_TABI, v23c.CONVERSION_TYPE_TAB):
                nr = self.ref_param_nr

                size = v23c.CC_COMMON_BLOCK_SIZE + nr * 16

                if block_size == v23c.MAX_UINT16:
                    stream.seek(address)
                    raw_bytes = stream.read(size)
                    conversion = ChannelConversion(raw_bytes=raw_bytes, stream=stream, address=address)
                    conversion.block_len = size

                    # self.update(conversion)
                    self.referenced_blocks = conversion.referenced_blocks

                else:
                    float_values: tuple[float, ...] = unpack_from(f"<{2 * nr}d", block, v23c.CC_COMMON_SHORT_SIZE)
                    for i in range(nr):
                        (self[f"raw_{i}"], self[f"phys_{i}"]) = (
                            float_values[i * 2],
                            float_values[2 * i + 1],
                        )

            elif conv_type in (v23c.CONVERSION_TYPE_POLY, v23c.CONVERSION_TYPE_RAT):
                (self.P1, self.P2, self.P3, self.P4, self.P5, self.P6) = typing.cast(
                    tuple[float, float, float, float, float, float],
                    unpack_from("<6d", block, v23c.CC_COMMON_SHORT_SIZE),
                )

            elif conv_type in (v23c.CONVERSION_TYPE_EXPO, v23c.CONVERSION_TYPE_LOGH):
                (
                    self.P1,
                    self.P2,
                    self.P3,
                    self.P4,
                    self.P5,
                    self.P6,
                    self.P7,
                ) = typing.cast(
                    tuple[float, float, float, float, float, float, float],
                    unpack_from("<7d", block, v23c.CC_COMMON_SHORT_SIZE),
                )

            elif conv_type == v23c.CONVERSION_TYPE_TABX:
                nr = self.ref_param_nr

                size = v23c.CC_COMMON_BLOCK_SIZE + nr * 40

                if block_size == v23c.MAX_UINT16:
                    stream.seek(address)
                    raw_bytes = stream.read(size)
                    conversion = ChannelConversion(raw_bytes=raw_bytes, stream=stream, address=address)
                    conversion.block_len = size

                    for attr in get_fields(conversion):
                        setattr(self, attr, getattr(conversion, attr))

                    self.referenced_blocks = conversion.referenced_blocks

                else:
                    values = unpack_from("<" + "d32s" * nr, block, v23c.CC_COMMON_SHORT_SIZE)

                    for i in range(nr):
                        (self[f"param_val_{i}"], self[f"text_{i}"]) = (
                            values[i * 2],
                            values[2 * i + 1],
                        )

            elif conv_type == v23c.CONVERSION_TYPE_RTABX:
                nr = self.ref_param_nr - 1

                size = v23c.CC_COMMON_BLOCK_SIZE + (nr + 1) * 20

                if block_size == v23c.MAX_UINT16:
                    stream.seek(address)
                    raw_bytes = stream.read(size)
                    conversion = ChannelConversion(raw_bytes=raw_bytes, stream=stream, address=address)
                    conversion.block_len = size

                    for attr in get_fields(conversion):
                        setattr(self, attr, getattr(conversion, attr))
                    self.referenced_blocks = conversion.referenced_blocks

                else:
                    (
                        self.default_lower,
                        self.default_upper,
                        self.default_addr,
                    ) = typing.cast(tuple[float, float, int], unpack_from("<2dI", block, v23c.CC_COMMON_SHORT_SIZE))

                    if self.default_addr:
                        self.referenced_blocks["default_addr"] = get_text_v3(
                            address=self.default_addr,
                            stream=stream,
                            mapped=mapped,
                            decode=False,
                        )
                    else:
                        self.referenced_blocks["default_addr"] = b""

                    values = unpack_from("<" + "2dI" * nr, block, v23c.CC_COMMON_SHORT_SIZE + 20)

                    for i in range(nr):
                        (self[f"lower_{i}"], self[f"upper_{i}"], self[f"text_{i}"]) = (
                            values[i * 3],
                            values[3 * i + 1],
                            values[3 * i + 2],
                        )
                        if values[3 * i + 2]:
                            block = get_text_v3(
                                address=values[3 * i + 2],
                                stream=stream,
                                mapped=mapped,
                                decode=False,
                            )
                            self.referenced_blocks[f"text_{i}"] = block

                        else:
                            self.referenced_blocks[f"text_{i}"] = b""

            if self.id != b"CC":
                message = f'Expected "CC" block @{hex(address)} but found "{self.id!r}"'

                logger.exception(message)
                raise MdfException(message)

        else:
            self.address = 0
            self.id = b"CC"
            self.unit_field = kwargs.get("unit", b"")

            if kwargs["conversion_type"] == v23c.CONVERSION_TYPE_NONE:
                self.block_len = v23c.CC_COMMON_BLOCK_SIZE
                self.range_flag = kwargs.get("range_flag", 1)
                self.min_phy_value = kwargs.get("min_phy_value", 0)
                self.max_phy_value = kwargs.get("max_phy_value", 0)
                self.unit_field = kwargs.get("unit", ("\0" * 20).encode("latin-1"))
                self.conversion_type = v23c.CONVERSION_TYPE_NONE
                self.ref_param_nr = 0

            elif kwargs["conversion_type"] == v23c.CONVERSION_TYPE_LINEAR:
                self.block_len = v23c.CC_LIN_BLOCK_SIZE
                self.range_flag = kwargs.get("range_flag", 1)
                self.min_phy_value = kwargs.get("min_phy_value", 0)
                self.max_phy_value = kwargs.get("max_phy_value", 0)
                self.unit_field = kwargs.get("unit", ("\0" * 20).encode("latin-1"))
                self.conversion_type = v23c.CONVERSION_TYPE_LINEAR
                self.ref_param_nr = 2
                self.b = kwargs.get("b", 0)
                self.a = kwargs.get("a", 1)
                if self.block_len != v23c.CC_LIN_BLOCK_SIZE:
                    self.CANapeHiddenExtra = kwargs["CANapeHiddenExtra"]

            elif kwargs["conversion_type"] in (
                v23c.CONVERSION_TYPE_POLY,
                v23c.CONVERSION_TYPE_RAT,
            ):
                self.block_len = v23c.CC_POLY_BLOCK_SIZE
                self.range_flag = kwargs.get("range_flag", 1)
                self.min_phy_value = kwargs.get("min_phy_value", 0)
                self.max_phy_value = kwargs.get("max_phy_value", 0)
                self.unit_field = kwargs.get("unit", ("\0" * 20).encode("latin-1"))
                self.conversion_type = kwargs["conversion_type"]
                self.ref_param_nr = 6
                self.P1 = kwargs.get("P1", 0)
                self.P2 = kwargs.get("P2", 0)
                self.P3 = kwargs.get("P3", 0)
                self.P4 = kwargs.get("P4", 0)
                self.P5 = kwargs.get("P5", 0)
                self.P6 = kwargs.get("P6", 0)

            elif kwargs["conversion_type"] in (
                v23c.CONVERSION_TYPE_EXPO,
                v23c.CONVERSION_TYPE_LOGH,
            ):
                self.block_len = v23c.CC_EXPO_BLOCK_SIZE
                self.range_flag = kwargs.get("range_flag", 1)
                self.min_phy_value = kwargs.get("min_phy_value", 0)
                self.max_phy_value = kwargs.get("max_phy_value", 0)
                self.unit_field = kwargs.get("unit", ("\0" * 20).encode("latin-1"))
                self.conversion_type = kwargs["conversion_type"]
                self.ref_param_nr = 7
                self.P1 = kwargs.get("P1", 0)
                self.P2 = kwargs.get("P2", 0)
                self.P3 = kwargs.get("P3", 0)
                self.P4 = kwargs.get("P4", 0)
                self.P5 = kwargs.get("P5", 0)
                self.P6 = kwargs.get("P6", 0)
                self.P7 = kwargs.get("P7", 0)

            elif kwargs["conversion_type"] == v23c.CONVERSION_TYPE_FORMULA:
                formula = kwargs["formula"]
                formula_len = len(formula)
                if isinstance(formula, bytes):
                    self.formula = formula.decode("latin-1")
                    formula += b"\0"
                else:
                    self.formula = formula
                    formula = formula.encode("latin-1") + b"\0"
                self.block_len = 46 + formula_len + 1
                self.range_flag = kwargs.get("range_flag", 1)
                self.min_phy_value = kwargs.get("min_phy_value", 0)
                self.max_phy_value = kwargs.get("max_phy_value", 0)
                self.unit_field = kwargs.get("unit", ("\0" * 20).encode("latin-1"))
                self.conversion_type = v23c.CONVERSION_TYPE_FORMULA
                self.ref_param_nr = formula_len
                self.formula_field = formula

            elif kwargs["conversion_type"] in (
                v23c.CONVERSION_TYPE_TABI,
                v23c.CONVERSION_TYPE_TAB,
            ):
                nr = kwargs["ref_param_nr"]
                self.block_len = v23c.CC_COMMON_BLOCK_SIZE + nr * 2 * 8
                self.range_flag = kwargs.get("range_flag", 1)
                self.min_phy_value = kwargs.get("min_phy_value", 0)
                self.max_phy_value = kwargs.get("max_phy_value", 0)
                self.unit_field = kwargs.get("unit", ("\0" * 20).encode("latin-1"))
                self.conversion_type = kwargs["conversion_type"]
                self.ref_param_nr = nr
                for i in range(nr):
                    self[f"raw_{i}"] = kwargs[f"raw_{i}"]  # type: ignore[literal-required]
                    self[f"phys_{i}"] = kwargs[f"phys_{i}"]  # type: ignore[literal-required]

            elif kwargs["conversion_type"] == v23c.CONVERSION_TYPE_TABX:
                nr = kwargs["ref_param_nr"]
                self.block_len = v23c.CC_COMMON_BLOCK_SIZE + 40 * nr
                self.range_flag = kwargs.get("range_flag", 0)
                self.min_phy_value = kwargs.get("min_phy_value", 0)
                self.max_phy_value = kwargs.get("max_phy_value", 0)
                self.unit_field = kwargs.get("unit", ("\0" * 20).encode("latin-1"))
                self.conversion_type = v23c.CONVERSION_TYPE_TABX
                self.ref_param_nr = nr

                for i in range(nr):
                    self[f"param_val_{i}"] = kwargs[f"param_val_{i}"]  # type: ignore[literal-required]
                    self[f"text_{i}"] = kwargs[f"text_{i}"]  # type: ignore[literal-required]

            elif kwargs["conversion_type"] == v23c.CONVERSION_TYPE_RTABX:
                nr = kwargs["ref_param_nr"]
                self.block_len = v23c.CC_COMMON_BLOCK_SIZE + 20 * nr
                self.range_flag = kwargs.get("range_flag", 0)
                self.min_phy_value = kwargs.get("min_phy_value", 0)
                self.max_phy_value = kwargs.get("max_phy_value", 0)
                self.unit_field = kwargs.get("unit", ("\0" * 20).encode("latin-1"))
                self.conversion_type = v23c.CONVERSION_TYPE_RTABX
                self.ref_param_nr = nr

                self.default_lower = 0
                self.default_upper = 0
                self.default_addr = 0
                key = "default_addr"
                if key in kwargs:
                    self.referenced_blocks[key] = kwargs[key]  # type: ignore[literal-required]
                else:
                    self.referenced_blocks[key] = b""

                for i in range(nr - 1):
                    self[f"lower_{i}"] = kwargs[f"lower_{i}"]  # type: ignore[literal-required]
                    self[f"upper_{i}"] = kwargs[f"upper_{i}"]  # type: ignore[literal-required]
                    key = f"text_{i}"
                    self.default_addr = 0
                    self.referenced_blocks[key] = kwargs[key]  # type: ignore[literal-required]
            else:
                message = f'Conversion type "{kwargs["conversion_type"]}" not implemented'
                logger.exception(message)
                raise MdfException(message)

    def to_blocks(
        self,
        address: int,
        blocks: list[bytes | SupportsBytes],
        defined_texts: dict[bytes | str, int],
        cc_map: dict[bytes, int],
    ) -> int:
        self.unit_field = self.unit.encode("latin-1", "ignore")[:19]

        if self.conversion_type == v23c.CONVERSION_TYPE_FORMULA:
            formula = self.formula
            if not formula.endswith("\0"):
                formula += "\0"
            self.formula_field = formula.encode("latin-1")
            self.block_len = v23c.CC_COMMON_BLOCK_SIZE + len(self.formula_field)

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
                        tx_block = TextBlock(text=text)
                        defined_texts[text] = address
                        blocks.append(tx_block)
                        self[key] = address
                        address += tx_block.block_len
            else:
                self[key] = 0

        bts = bytes(self)
        if bts in cc_map:
            self.address = cc_map[bts]
        else:
            blocks.append(bts)
            self.address = address
            cc_map[bts] = address
            size = self.block_len
            address += len(bts)

        return address

    def metadata(self, indent: str = "") -> str:
        conv = self.conversion_type
        keys: Sequence[str]
        if conv == v23c.CONVERSION_TYPE_NONE:
            keys = v23c.KEYS_CONVERSION_NONE
        elif conv == v23c.CONVERSION_TYPE_FORMULA:
            keys = v23c.KEYS_CONVERSION_FORMULA
        elif conv == v23c.CONVERSION_TYPE_LINEAR:
            keys = v23c.KEYS_CONVERSION_LINEAR
            if self.block_len != v23c.CC_LIN_BLOCK_SIZE:
                keys += ("CANapeHiddenExtra",)

                nr = self.ref_param_nr
                new_keys: list[str] = []
                for i in range(nr):
                    new_keys.extend((f"param_val_{i}", f"text_{i}"))
                keys += tuple(new_keys)

        elif conv in (v23c.CONVERSION_TYPE_POLY, v23c.CONVERSION_TYPE_RAT):
            keys = v23c.KEYS_CONVERSION_POLY_RAT
        elif conv in (v23c.CONVERSION_TYPE_EXPO, v23c.CONVERSION_TYPE_LOGH):
            keys = v23c.KEYS_CONVERSION_EXPO_LOGH
        elif conv in (v23c.CONVERSION_TYPE_TABI, v23c.CONVERSION_TYPE_TAB):
            nr = self.ref_param_nr
            keys = list(v23c.KEYS_CONVERSION_NONE)
            for i in range(nr):
                keys.append(f"raw_{i}")
                keys.append(f"phys_{i}")
        elif conv == v23c.CONVERSION_TYPE_RTABX:
            nr = self.ref_param_nr
            keys = list(v23c.KEYS_CONVERSION_NONE)
            keys += ["default_lower", "default_upper", "default_addr"]
            for i in range(nr - 1):
                keys.append(f"lower_{i}")
                keys.append(f"upper_{i}")
                keys.append(f"text_{i}")
        elif conv == v23c.CONVERSION_TYPE_TABX:
            nr = self.ref_param_nr
            keys = list(v23c.KEYS_CONVERSION_NONE)
            for i in range(nr):
                keys.append(f"param_val_{i}")
                keys.append(f"text_{i}")

        max_len = max(len(key) for key in keys)
        template = f"{{: <{max_len}}}: {{}}"

        metadata: list[str] = []
        lines = f"""
address: {hex(self.address)}

""".split("\n")

        for key in keys:
            val = getattr(self, key)
            if key.endswith("addr") or (key.startswith("text_") and isinstance(val, int)):
                lines.append(template.format(key, hex(val)))
            elif isinstance(val, float):
                lines.append(template.format(key, val))
            else:
                if isinstance(val, bytes):
                    lines.append(template.format(key, val.strip(b"\0")))
                else:
                    lines.append(template.format(key, val))

            if key == "conversion_type":
                lines[-1] += f" [{v23c.CONVERSION_TYPE_TO_STRING[self.conversion_type]}]"
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

    @overload
    def convert(
        self,
        values: NDArray[Any],
        as_object: bool = ...,
        as_bytes: bool = ...,
        ignore_value2text_conversions: bool = ...,
    ) -> NDArray[Any]: ...

    @overload
    def convert(
        self,
        values: ArrayLike,
        as_object: bool = ...,
        as_bytes: bool = ...,
        ignore_value2text_conversions: bool = ...,
    ) -> NDArray[Any] | np.number[Any]: ...

    def convert(
        self,
        values: ArrayLike,
        as_object: bool = False,
        as_bytes: bool = False,
        ignore_value2text_conversions: bool = False,
    ) -> NDArray[Any] | np.number[Any]:
        conversion_type = self.conversion_type
        scalar = False

        if not isinstance(values, np.ndarray):
            if isinstance(values, (int, float)):
                new_values = np.array([values])
                scalar = True
            else:
                new_values = np.array(values)
        else:
            new_values = values

        if conversion_type == v23c.CONVERSION_TYPE_NONE:
            pass

        elif conversion_type == v23c.CONVERSION_TYPE_LINEAR:
            a = self.a
            b = self.b
            if (a, b) != (1, 0):
                new_values = new_values * a
                if b:
                    new_values += b

        elif conversion_type in (v23c.CONVERSION_TYPE_TABI, v23c.CONVERSION_TYPE_TAB):
            nr = self.ref_param_nr

            raw_vals = np.array([self[f"raw_{i}"] for i in range(nr)])
            phys = np.array([self[f"phys_{i}"] for i in range(nr)])

            if conversion_type == v23c.CONVERSION_TYPE_TABI:
                new_values = np.interp(new_values, raw_vals, phys)
            else:
                dim = raw_vals.shape[0]

                inds = np.searchsorted(raw_vals, new_values)

                inds[inds >= dim] = dim - 1  # type: ignore[index, unused-ignore]

                inds2 = inds - 1
                inds2[inds2 < 0] = 0  # type: ignore[index, unused-ignore]

                cond = np.abs(new_values - raw_vals[inds]) >= np.abs(new_values - raw_vals[inds2])

                new_values = np.where(cond, phys[inds2], phys[inds])

        elif conversion_type == v23c.CONVERSION_TYPE_TABX:
            if not ignore_value2text_conversions:
                nr = self.ref_param_nr
                raw_vals = np.array([self[f"param_val_{i}"] for i in range(nr)])
                phys = np.array([self[f"text_{i}"] for i in range(nr)])

                x = sorted(zip(raw_vals, phys, strict=False))
                raw_vals = np.array([e[0] for e in x], dtype="<i8")
                phys = np.array([e[1] for e in x])

                default = b""

                idx1 = np.searchsorted(raw_vals, new_values, side="right") - 1
                idx2 = np.searchsorted(raw_vals, new_values, side="left")

                idx = np.argwhere(idx1 != idx2).flatten()

                new_values = np.zeros(len(new_values), dtype=max(phys.dtype, np.array([default]).dtype))

                new_values[idx] = default
                idx = np.argwhere(idx1 == idx2).flatten()
                if len(idx):
                    new_values[idx] = phys[idx1[idx]]  # type: ignore[index, unused-ignore]

        elif conversion_type == v23c.CONVERSION_TYPE_RTABX:
            if not ignore_value2text_conversions:
                nr = self.ref_param_nr - 1

                phys_list: list[bytes | ChannelConversion | None] = []
                for i in range(nr):
                    value = self.referenced_blocks[f"text_{i}"]
                    phys_list.append(value)

                phys = np.array(phys_list)

                default = typing.cast(bytes, self.referenced_blocks["default_addr"])

                if b"{X}" in default:
                    default_text = default.decode("latin-1").replace("{X}", "X").split('"')[1]
                else:
                    default_text = None

                lower = np.array([self[f"lower_{i}"] for i in range(nr)])
                upper = np.array([self[f"upper_{i}"] for i in range(nr)])

                idx1 = np.searchsorted(lower, new_values, side="right") - 1
                idx2 = np.searchsorted(upper, new_values, side="left")

                idx = np.argwhere(idx1 != idx2).flatten()

                if default_text and len(idx):
                    X = new_values[idx]
                    new_values = np.zeros(len(new_values), dtype=np.float64)

                    a = float(default_text.split("*")[0])
                    b = float(default_text.split("X")[-1])
                    new_values[idx] = a * X + b

                    idx = np.argwhere(idx1 == idx2).flatten()
                    if len(idx):
                        new_values[idx] = np.nan

                else:
                    if len(idx):
                        new_values = np.zeros(len(new_values), dtype=max(phys.dtype, np.array([default]).dtype))
                        new_values[idx] = default

                        idx = np.argwhere(idx1 == idx2).flatten()
                        if len(idx):
                            new_values[idx] = phys[idx1[idx]]  # type: ignore[index, unused-ignore]
                    else:
                        new_values = phys[idx1]

        elif conversion_type in (v23c.CONVERSION_TYPE_EXPO, v23c.CONVERSION_TYPE_LOGH):
            # pylint: disable=C0103

            func: np.ufunc
            if conversion_type == v23c.CONVERSION_TYPE_EXPO:
                func = np.exp
            else:
                func = np.log
            P1 = self.P1
            P2 = self.P2
            P3 = self.P3
            P4 = self.P4
            P5 = self.P5
            P6 = self.P6
            P7 = self.P7
            if P4 == 0:
                new_values = func(((new_values - P7) * P6 - P3) / P1) / P2
            elif P1 == 0:
                new_values = func((P3 / (new_values - P7) - P6) / P4) / P5
            else:
                message = f"wrong conversion {conversion_type}"
                raise ValueError(message)

        elif conversion_type == v23c.CONVERSION_TYPE_RAT:
            # pylint: disable=unused-variable,C0103

            P1 = self.P1
            P2 = self.P2
            P3 = self.P3
            P4 = self.P4
            P5 = self.P5
            P6 = self.P6

            X = new_values
            if (P1, P3, P4, P5) == (0, 0, 0, 0):
                if P2 != P6:
                    new_values = new_values * (P2 / P6)

            elif (P2, P3, P4, P6) == (0, 0, 0, 0):
                if P1 != P5:
                    new_values = new_values * (P1 / P5)
            else:
                try:
                    new_values = evaluate(v23c.RAT_CONV_TEXT)
                except TypeError:
                    new_values = (P1 * X**2 + P2 * X + P3) / (P4 * X**2 + P5 * X + P6)

        elif conversion_type == v23c.CONVERSION_TYPE_POLY:
            # pylint: disable=unused-variable,C0103

            P1 = self.P1
            P2 = self.P2
            P3 = self.P3
            P4 = self.P4
            P5 = self.P5
            P6 = self.P6

            X = new_values

            coefs = (P2, P3, P5, P6)
            if coefs == (0, 0, 0, 0):
                if P1 != P4:
                    try:
                        new_values = evaluate(v23c.POLY_CONV_SHORT_TEXT)
                    except TypeError:
                        new_values = P4 * X / P1
            else:
                try:
                    new_values = evaluate(v23c.POLY_CONV_LONG_TEXT)
                except TypeError:
                    new_values = (P2 - (P4 * (X - P5 - P6))) / (P3 * (X - P5 - P6) - P1)

        elif conversion_type == v23c.CONVERSION_TYPE_FORMULA:
            # pylint: disable=unused-variable,C0103

            try:
                X1 = new_values
                INF = np.inf
                NaN = np.nan
                new_values = evaluate(self.formula)
            except:
                if lambdify is not None and symbols is not None:
                    X_symbol = symbols("X1")
                    expr = lambdify(
                        X_symbol,
                        self.formula,
                        modules=[{"INF": np.inf, "NaN": np.nan}, "numpy"],
                        dummify=False,
                        cse=True,
                    )
                    new_values = expr(new_values)

        if scalar:
            return typing.cast(np.number[Any], new_values[0])
        else:
            return new_values

    def __getitem__(self, item: str) -> object:
        return self.__getattribute__(item)

    def __setitem__(self, item: str, value: object) -> None:
        self.__setattr__(item, value)

    def __bytes__(self) -> bytes:
        conv = self.conversion_type

        # compute the fmt
        if conv == v23c.CONVERSION_TYPE_NONE:
            fmt = v23c.FMT_CONVERSION_COMMON
        elif conv == v23c.CONVERSION_TYPE_FORMULA:
            fmt = v23c.FMT_CONVERSION_FORMULA.format(self.block_len - v23c.CC_COMMON_BLOCK_SIZE)
        elif conv == v23c.CONVERSION_TYPE_LINEAR:
            fmt = v23c.FMT_CONVERSION_LINEAR
            if self.block_len != v23c.CC_LIN_BLOCK_SIZE:
                fmt += f"{self.block_len - v23c.CC_LIN_BLOCK_SIZE}s"
        elif conv in (v23c.CONVERSION_TYPE_POLY, v23c.CONVERSION_TYPE_RAT):
            fmt = v23c.FMT_CONVERSION_POLY_RAT
        elif conv in (v23c.CONVERSION_TYPE_EXPO, v23c.CONVERSION_TYPE_LOGH):
            fmt = v23c.FMT_CONVERSION_EXPO_LOGH
        elif conv in (v23c.CONVERSION_TYPE_TABI, v23c.CONVERSION_TYPE_TAB):
            nr = self.ref_param_nr
            fmt = v23c.FMT_CONVERSION_COMMON + f"{2 * nr}d"
        elif conv == v23c.CONVERSION_TYPE_RTABX:
            nr = self.ref_param_nr
            fmt = v23c.FMT_CONVERSION_COMMON + "2dI" * nr
        elif conv == v23c.CONVERSION_TYPE_TABX:
            nr = self.ref_param_nr
            fmt = v23c.FMT_CONVERSION_COMMON + "d32s" * nr

        keys: Sequence[str]
        if conv == v23c.CONVERSION_TYPE_NONE:
            keys = v23c.KEYS_CONVERSION_NONE
        elif conv == v23c.CONVERSION_TYPE_FORMULA:
            keys = v23c.KEYS_CONVERSION_FORMULA
        elif conv == v23c.CONVERSION_TYPE_LINEAR:
            keys = v23c.KEYS_CONVERSION_LINEAR
            if self.block_len != v23c.CC_LIN_BLOCK_SIZE:
                keys += ("CANapeHiddenExtra",)
        elif conv in (v23c.CONVERSION_TYPE_POLY, v23c.CONVERSION_TYPE_RAT):
            keys = v23c.KEYS_CONVERSION_POLY_RAT
        elif conv in (v23c.CONVERSION_TYPE_EXPO, v23c.CONVERSION_TYPE_LOGH):
            keys = v23c.KEYS_CONVERSION_EXPO_LOGH
        elif conv in (v23c.CONVERSION_TYPE_TABI, v23c.CONVERSION_TYPE_TAB):
            nr = self.ref_param_nr
            keys = list(v23c.KEYS_CONVERSION_NONE)
            for i in range(nr):
                keys.append(f"raw_{i}")
                keys.append(f"phys_{i}")
        elif conv == v23c.CONVERSION_TYPE_RTABX:
            nr = self.ref_param_nr
            keys = list(v23c.KEYS_CONVERSION_NONE)
            keys += ["default_lower", "default_upper", "default_addr"]
            for i in range(nr - 1):
                keys.append(f"lower_{i}")
                keys.append(f"upper_{i}")
                keys.append(f"text_{i}")
        elif conv == v23c.CONVERSION_TYPE_TABX:
            nr = self.ref_param_nr
            keys = list(v23c.KEYS_CONVERSION_NONE)
            for i in range(nr):
                keys.append(f"param_val_{i}")
                keys.append(f"text_{i}")

        self.block_len = min(self.block_len, v23c.MAX_UINT16)
        result = pack(fmt, *[self[key] for key in keys])
        return result

    def __str__(self) -> str:
        fields = []
        for attr in dir(self):
            if attr[:2] + attr[-2:] == "____":
                continue
            try:
                if callable(getattr(self, attr)):
                    continue
                fields.append(f"{attr}:{getattr(self, attr)}")
            except AttributeError:
                continue
        return f"ChannelConversion (referenced blocks: {self.referenced_blocks}, address: {hex(self.address)}, fields: {fields})"


class ChannelDependencyKwargs(BlockKwargs, total=False):
    sd_nr: int


class ChannelDependency:
    """CDBLOCK class.

    CDBLOCK fields:

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

    Other attributes:

    * ``address`` - int : block address inside MDF file
    * ``referenced_channels`` - list : list of (group index, channel index)
      pairs

    Parameters
    ----------
    stream : file handle
        MDF file handle.
    address : int
        Block address inside MDF file.
    for dynamically created objects :
        See the key-value pairs.
    """

    def __init__(self, **kwargs: Unpack[ChannelDependencyKwargs]) -> None:
        super().__init__()

        self.referenced_channels: list[tuple[int, int]] = []

        try:
            stream = kwargs["stream"]
            self.address = address = kwargs["address"]
            stream.seek(address)

            (self.id, self.block_len, self.dependency_type, self.sd_nr) = typing.cast(
                tuple[bytes, int, int, int], unpack("<2s3H", stream.read(8))
            )

            links_size = 3 * 4 * self.sd_nr
            links: tuple[int, ...] = unpack(f"<{3 * self.sd_nr}I", stream.read(links_size))

            for i in range(self.sd_nr):
                self[f"dg_{i}"] = links[3 * i]
                self[f"cg_{i}"] = links[3 * i + 1]
                self[f"ch_{i}"] = links[3 * i + 2]

            optional_dims_nr = (self.block_len - 8 - links_size) // 2
            if optional_dims_nr:
                dims: tuple[int, ...] = unpack(f"<{optional_dims_nr}H", stream.read(optional_dims_nr * 2))
                for i, dim in enumerate(dims):
                    self[f"dim_{i}"] = dim

            if self.id != b"CD":
                message = f'Expected "CD" block @{hex(address)} but found "{self.id!r}"'

                logger.exception(message)
                raise MdfException(message)

        except KeyError:
            sd_nr = kwargs["sd_nr"]
            self.id = b"CD"
            self.block_len = 8 + 3 * 4 * sd_nr
            self.dependency_type = 1
            self.sd_nr = sd_nr
            for i in range(sd_nr):
                self[f"dg_{i}"] = 0
                self[f"cg_{i}"] = 0
                self[f"ch_{i}"] = 0
            i = 0
            while True:
                try:
                    self[f"dim_{i}"] = kwargs[f"dim_{i}"]  # type: ignore[literal-required]
                    i += 1
                except KeyError:
                    break
            if i:
                self.dependency_type = 256 + i
                self.block_len += 2 * i

    def __getitem__(self, item: str) -> object:
        return self.__getattribute__(item)

    def __setitem__(self, item: str, value: object) -> None:
        self.__setattr__(item, value)

    def __bytes__(self) -> bytes:
        fmt = f"<2s3H{self.sd_nr * 3}I"
        keys: tuple[str, ...] = ("id", "block_len", "dependency_type", "sd_nr")
        for i in range(self.sd_nr):
            keys += (f"dg_{i}", f"cg_{i}", f"ch_{i}")
        links_size = 3 * 4 * self.sd_nr
        option_dims_nr = (self.block_len - 8 - links_size) // 2
        if option_dims_nr:
            fmt += f"{option_dims_nr}H"
            keys += tuple(f"dim_{i}" for i in range(option_dims_nr))
        result = pack(fmt, *[self[key] for key in keys])
        return result


class ChannelExtensionKwargs(BlockKwargs, total=False):
    raw_bytes: bytes
    block_len: int
    type: int
    module_nr: int
    module_address: int
    description: bytes
    ECU_identification: bytes
    reserved0: bytes
    CAN_id: int
    CAN_ch_index: int
    message_name: bytes
    sender_name: bytes


class ChannelExtension:
    """CEBLOCK class.

    CEBLOCK has the following common fields:

    * ``id`` - bytes : block ID; always b'CE'
    * ``block_len`` - int : block bytes size
    * ``type`` - int : extension type identifier

    CEBLOCK has the following specific fields:

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
      * ``sender_name`` - bytes : sender name
      * ``reserved0`` - bytes : reserved bytes

    Other attributes:

    * ``address`` - int : block address inside MDF file
    * ``comment`` - str : extension comment
    * ``name`` - str : extension name
    * ``path`` - str : extension path

    Parameters
    ----------
    stream : file handle
        MDF file handle.
    address : int
        Block address inside MDF file.
    for dynamically created objects :
        See the key-value pairs.
    """

    __slots__ = (
        "CAN_ch_index",
        "CAN_id",
        "ECU_identification",
        "address",
        "block_len",
        "comment",
        "description",
        "id",
        "message_name",
        "module_address",
        "module_nr",
        "name",
        "path",
        "reserved0",
        "sender_name",
        "type",
    )

    def __init__(self, **kwargs: Unpack[ChannelExtensionKwargs]) -> None:
        super().__init__()

        self.name = self.path = self.comment = ""

        if "stream" in kwargs:
            stream = kwargs["stream"]
            try:
                (self.id, self.block_len, self.type) = SOURCE_COMMON_uf(kwargs["raw_bytes"])
                if self.type == v23c.SOURCE_ECU:
                    (
                        self.module_nr,
                        self.module_address,
                        self.description,
                        self.ECU_identification,
                        self.reserved0,
                    ) = SOURCE_EXTRA_ECU_uf(kwargs["raw_bytes"], 6)
                elif self.type == v23c.SOURCE_VECTOR:
                    (
                        self.CAN_id,
                        self.CAN_ch_index,
                        self.message_name,
                        self.sender_name,
                        self.reserved0,
                    ) = SOURCE_EXTRA_VECTOR_uf(kwargs["raw_bytes"], 6)

                self.address = kwargs.get("address", 0)
            except KeyError:
                if utils.stream_is_mmap(stream, kwargs.get("mapped", False)):
                    self.address = address = kwargs["address"]

                    (self.id, self.block_len, self.type) = SOURCE_COMMON_uf(stream, address)

                    if self.type == v23c.SOURCE_ECU:
                        (
                            self.module_nr,
                            self.module_address,
                            self.description,
                            self.ECU_identification,
                            self.reserved0,
                        ) = SOURCE_EXTRA_ECU_uf(stream, address + 6)
                    elif self.type == v23c.SOURCE_VECTOR:
                        (
                            self.CAN_id,
                            self.CAN_ch_index,
                            self.message_name,
                            self.sender_name,
                            self.reserved0,
                        ) = SOURCE_EXTRA_VECTOR_uf(stream, address + 6)
                else:
                    self.address = address = kwargs["address"]
                    stream.seek(address)
                    block = stream.read(v23c.CE_BLOCK_SIZE)
                    (self.id, self.block_len, self.type) = SOURCE_COMMON_uf(block)

                    if self.type == v23c.SOURCE_ECU:
                        (
                            self.module_nr,
                            self.module_address,
                            self.description,
                            self.ECU_identification,
                            self.reserved0,
                        ) = SOURCE_EXTRA_ECU_uf(block, 6)
                    elif self.type == v23c.SOURCE_VECTOR:
                        (
                            self.CAN_id,
                            self.CAN_ch_index,
                            self.message_name,
                            self.sender_name,
                            self.reserved0,
                        ) = SOURCE_EXTRA_VECTOR_uf(block, 6)

            if self.id != b"CE":
                message = f'Expected "CE" block @{hex(address)} but found "{self.id!r}"'

                logger.exception(message)
                raise MdfException(message)

        else:
            self.address = 0
            self.id = b"CE"
            self.block_len = kwargs.get("block_len", v23c.CE_BLOCK_SIZE)
            self.type = kwargs.get("type", 2)
            if self.type == v23c.SOURCE_ECU:
                self.module_nr = kwargs.get("module_nr", 0)
                self.module_address = kwargs.get("module_address", 0)
                self.description = kwargs.get("description", b"\0")
                self.ECU_identification = kwargs.get("ECU_identification", b"\0")
                self.reserved0 = kwargs.get("reserved0", b"\0")
            elif self.type == v23c.SOURCE_VECTOR:
                self.CAN_id = kwargs.get("CAN_id", 0)
                self.CAN_ch_index = kwargs.get("CAN_ch_index", 0)
                self.message_name = kwargs.get("message_name", b"\0")
                self.sender_name = kwargs.get("sender_name", b"\0")
                self.reserved0 = kwargs.get("reserved0", b"\0")

        if self.type == v23c.SOURCE_ECU:
            self.path = self.ECU_identification.decode("latin-1").strip(" \t\n\r\0")
            self.name = self.description.decode("latin-1").strip(" \t\n\r\0")
            self.comment = f"Module number={self.module_nr} @ address={self.module_address}"
        else:
            self.path = self.sender_name.decode("latin-1").strip(" \t\n\r\0")
            self.name = self.message_name.decode("latin-1").strip(" \t\n\r\0")
            self.comment = f"Message ID={hex(self.CAN_id)} on CAN bus {self.CAN_ch_index}"

    def to_blocks(
        self,
        address: int,
        blocks: list[bytes | SupportsBytes],
        defined_texts: dict[bytes | str, int],
        cc_map: dict[bytes, int],
    ) -> int:
        if self.type == v23c.SOURCE_ECU:
            self.ECU_identification = self.path.encode("latin-1")[:31]
            self.description = self.name.encode("latin-1")[:79]
        else:
            self.sender_name = self.path.encode("latin-1")[:35]
            self.message_name = self.name.encode("latin-1")[:35]

        bts = bytes(self)
        if bts in cc_map:
            self.address = cc_map[bts]
        else:
            blocks.append(bts)
            self.address = address
            cc_map[bts] = address
            address += self.block_len

        return address

    def __getitem__(self, item: str) -> object:
        return self.__getattribute__(item)

    def __setitem__(self, item: str, value: object) -> None:
        self.__setattr__(item, value)

    def metadata(self) -> str:
        if self.type == v23c.SOURCE_ECU:
            keys = (
                "id",
                "block_len",
                "type",
                "module_nr",
                "module_address",
                "description",
                "ECU_identification",
                "reserved0",
            )
        else:
            keys = (
                "id",
                "block_len",
                "type",
                "CAN_id",
                "CAN_ch_index",
                "message_name",
                "sender_name",
                "reserved0",
            )
        max_len = max(len(key) for key in keys)
        template = f"{{: <{max_len}}}: {{}}"

        metadata: list[str] = []
        lines = f"""
address: {hex(self.address)}

""".split("\n")

        for key in keys:
            val = getattr(self, key)
            if key.endswith("addr") or key.startswith("text_"):
                lines.append(template.format(key, hex(val)))
            elif isinstance(val, float):
                lines.append(template.format(key, val))
            else:
                if isinstance(val, bytes):
                    lines.append(template.format(key, val.strip(b"\0")))
                else:
                    lines.append(template.format(key, val))

            if key == "type":
                lines[-1] += f" = {v23c.SOURCE_TYPE_TO_STRING[self.type]}"
            # elif key == "bus_type":
            #     lines[-1] += f" = {v23c.BUS_TYPE_TO_STRING[self.bus_type]}"

        for line in lines:
            if not line:
                metadata.append(line)
            else:
                for wrapped_line in wrap(line, width=120):
                    metadata.append(wrapped_line)

        return "\n".join(metadata)

    def __bytes__(self) -> bytes:
        typ = self.type
        if typ == v23c.SOURCE_ECU:
            return v23c.SOURCE_ECU_p(
                self.id,
                self.block_len,
                self.type,
                self.module_nr,
                self.module_address,
                self.description,
                self.ECU_identification,
                self.reserved0,
            )
        else:
            return v23c.SOURCE_VECTOR_p(
                self.id,
                self.block_len,
                self.type,
                self.CAN_id,
                self.CAN_ch_index,
                self.message_name,
                self.sender_name,
                self.reserved0,
            )

    def __str__(self) -> str:
        fields = []
        for attr in dir(self):
            if attr[:2] + attr[-2:] == "____":
                continue
            try:
                if callable(getattr(self, attr)):
                    continue
                fields.append(f"{attr}:{getattr(self, attr)}")
            except AttributeError:
                continue
        return f"ChannelExtension (name: {self.name}, path: {self.path}, comment: {self.comment}, address: {hex(self.address)}, fields: {fields})"


class ChannelGroupKwargs(BlockKwargs, total=False):
    block_len: int
    next_cg_addr: int
    first_ch_addr: int
    comment_addr: int
    record_id: int
    ch_nr: int
    samples_byte_nr: int
    cycles_nr: int


class ChannelGroup:
    """CGBLOCK class.

    CGBLOCK fields:

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
    * ``sample_reduction_addr`` - int : address to first sample reduction
      block

    Other attributes:

    * ``address`` - int : block address inside MDF file
    * ``comment`` - str : channel group comment

    Parameters
    ----------
    stream : file handle
        MDF file handle.
    address : int
        Block address inside MDF file.
    for dynamically created objects :
        See the key-value pairs.

    Examples
    --------
    >>> with open('test.mdf', 'rb') as mdf:
    ...     cg1 = ChannelGroup(stream=mdf, address=0xBA52)
    >>> cg2 = ChannelGroup(sample_bytes_nr=32)
    >>> hex(cg1.address)
    0xBA52
    >>> cg1['id']
    b'CG'
    """

    __slots__ = (
        "address",
        "block_len",
        "ch_nr",
        "comment",
        "comment_addr",
        "cycles_nr",
        "first_ch_addr",
        "id",
        "next_cg_addr",
        "record_id",
        "sample_reduction_addr",
        "samples_byte_nr",
    )

    def __init__(self, **kwargs: Unpack[ChannelGroupKwargs]) -> None:
        super().__init__()
        self.comment = ""

        try:
            stream = kwargs["stream"]
            mapped = kwargs.get("mapped", False)
            self.address = address = kwargs["address"]

            if utils.stream_is_mmap(stream, mapped):
                (
                    self.id,
                    self.block_len,
                    self.next_cg_addr,
                    self.first_ch_addr,
                    self.comment_addr,
                    self.record_id,
                    self.ch_nr,
                    self.samples_byte_nr,
                    self.cycles_nr,
                ) = v23c.CHANNEL_GROUP_uf(stream, address)
                if self.block_len == v23c.CG_POST_330_BLOCK_SIZE:
                    # sample reduction blocks are not yet used
                    self.sample_reduction_addr = 0
            else:
                stream.seek(address)
                block = stream.read(v23c.CG_PRE_330_BLOCK_SIZE)

                (
                    self.id,
                    self.block_len,
                    self.next_cg_addr,
                    self.first_ch_addr,
                    self.comment_addr,
                    self.record_id,
                    self.ch_nr,
                    self.samples_byte_nr,
                    self.cycles_nr,
                ) = v23c.CHANNEL_GROUP_u(block)
                if self.block_len == v23c.CG_POST_330_BLOCK_SIZE:
                    # sample reduction blocks are not yet used
                    self.sample_reduction_addr = 0
            if self.id != b"CG":
                message = f'Expected "CG" block @{hex(address)} but found "{self.id!r}"'

                raise MdfException(message.format(self.id))
            if self.comment_addr:
                self.comment = get_text_v3(address=self.comment_addr, stream=stream, mapped=mapped)
        except KeyError:
            self.address = 0
            self.id = b"CG"
            self.block_len = kwargs.get("block_len", v23c.CG_PRE_330_BLOCK_SIZE)
            self.next_cg_addr = kwargs.get("next_cg_addr", 0)
            self.first_ch_addr = kwargs.get("first_ch_addr", 0)
            self.comment_addr = kwargs.get("comment_addr", 0)
            self.record_id = kwargs.get("record_id", 1)
            self.ch_nr = kwargs.get("ch_nr", 0)
            self.samples_byte_nr = kwargs.get("samples_byte_nr", 0)
            self.cycles_nr = kwargs.get("cycles_nr", 0)
            if self.block_len == v23c.CG_POST_330_BLOCK_SIZE:
                self.sample_reduction_addr = 0

    def to_blocks(
        self,
        address: int,
        blocks: list[bytes | SupportsBytes],
        defined_texts: dict[bytes | str, int],
        si_map: dict[bytes, int],
    ) -> int:
        text = self.comment
        if text:
            if text in defined_texts:
                self.comment_addr = defined_texts[text]
            else:
                tx_block = TextBlock(text=text)
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

    def __getitem__(self, item: str) -> object:
        return self.__getattribute__(item)

    def __setitem__(self, item: str, value: object) -> None:
        self.__setattr__(item, value)

    def __bytes__(self) -> bytes:
        if self.block_len == v23c.CG_POST_330_BLOCK_SIZE:
            return (
                v23c.CHANNEL_GROUP_p(
                    self.id,
                    self.block_len,
                    self.next_cg_addr,
                    self.first_ch_addr,
                    self.comment_addr,
                    self.record_id,
                    self.ch_nr,
                    self.samples_byte_nr,
                    self.cycles_nr,
                )
                + b"\0" * 4
            )
        else:
            return v23c.CHANNEL_GROUP_p(
                self.id,
                self.block_len,
                self.next_cg_addr,
                self.first_ch_addr,
                self.comment_addr,
                self.record_id,
                self.ch_nr,
                self.samples_byte_nr,
                self.cycles_nr,
            )

    def metadata(self) -> str:
        keys = (
            "id",
            "block_len",
            "next_cg_addr",
            "first_ch_addr",
            "comment_addr",
            "record_id",
            "ch_nr",
            "samples_byte_nr",
            "cycles_nr",
            "sample_reduction_addr",
        )

        max_len = max(len(key) for key in keys)
        template = f"{{: <{max_len}}}: {{}}"

        metadata: list[str] = []
        lines = f"""
address: {hex(self.address)}
comment: {self.comment}

""".split("\n")

        for key in keys:
            if not hasattr(self, key):
                continue
            val = getattr(self, key)
            if key.endswith("addr") or key.startswith("text_"):
                lines.append(template.format(key, hex(val)))
            elif isinstance(val, float):
                lines.append(template.format(key, val))
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


class DataBlockKwargs(BlockKwargs, total=False):
    size: int
    data: bytes


class DataBlock:
    """Data Block class (pseudo block not defined by the MDF 3 standard).

    `DataBlock` attributes:

    * ``data`` - bytes : raw samples bytes
    * ``address`` - int : block address

    Parameters
    ----------
    address : int
        Block address inside the measurement file.
    stream : file.io.handle
        Binary file stream.
    data : bytes
        When created dynamically.
    """

    __slots__ = "address", "data"

    def __init__(self, **kwargs: Unpack[DataBlockKwargs]) -> None:
        super().__init__()

        try:
            stream = kwargs["stream"]
            size = kwargs["size"]
            self.address = address = kwargs["address"]
            stream.seek(address)

            self.data = stream.read(size)

        except KeyError:
            self.address = 0
            self.data = kwargs.get("data", b"")

    def __getitem__(self, item: str) -> object:
        return self.__getattribute__(item)

    def __setitem__(self, item: str, value: object) -> None:
        self.__setattr__(item, value)

    def __bytes__(self) -> bytes:
        return self.data


class DataGroupKwargs(BlockKwargs, total=False):
    block_len: int
    next_dg_addr: int
    first_cg_addr: int
    comment_addr: int
    data_block_addr: int
    cg_nr: int
    record_id_len: int


class DataGroup:
    """DGBLOCK class.

    DGBLOCK fields:

    * ``id`` - bytes : block ID; always b'DG'
    * ``block_len`` - int : block bytes size
    * ``next_dg_addr`` - int : next DGBLOCK address
    * ``first_cg_addr`` - int : address of first channel group block (CGBLOCK)
    * ``trigger_addr`` - int : address of trigger block (TRBLOCK)
    * ``data_block_addr`` - address of data block
    * ``cg_nr`` - int : number of channel groups
    * ``record_id_len`` - int : number of record IDs in the data block
    * ``reserved0`` - bytes : reserved bytes

    Other attributes:

    * ``address`` - int : block address inside MDF file

    Parameters
    ----------
    stream : file handle
        MDF file handle.
    address : int
        Block address inside MDF file.
    for dynamically created objects :
        See the key-value pairs.
    """

    __slots__ = (
        "address",
        "block_len",
        "cg_nr",
        "data_block_addr",
        "first_cg_addr",
        "id",
        "next_dg_addr",
        "record_id_len",
        "reserved0",
        "trigger_addr",
    )

    def __init__(self, **kwargs: Unpack[DataGroupKwargs]) -> None:
        super().__init__()

        try:
            stream = kwargs["stream"]
            mapped = kwargs.get("mapped", False)
            self.address = address = kwargs["address"]
            if utils.stream_is_mmap(stream, mapped):
                block = stream.read(v23c.DG_PRE_320_BLOCK_SIZE)

                (
                    self.id,
                    self.block_len,
                    self.next_dg_addr,
                    self.first_cg_addr,
                    self.trigger_addr,
                    self.data_block_addr,
                    self.cg_nr,
                    self.record_id_len,
                ) = v23c.DATA_GROUP_PRE_320_uf(stream, address)

                if self.block_len == v23c.DG_POST_320_BLOCK_SIZE:
                    self.reserved0 = stream[
                        address + v23c.DG_PRE_320_BLOCK_SIZE : address + v23c.DG_POST_320_BLOCK_SIZE
                    ]
            else:
                stream.seek(address)
                block = stream.read(v23c.DG_PRE_320_BLOCK_SIZE)

                (
                    self.id,
                    self.block_len,
                    self.next_dg_addr,
                    self.first_cg_addr,
                    self.trigger_addr,
                    self.data_block_addr,
                    self.cg_nr,
                    self.record_id_len,
                ) = v23c.DATA_GROUP_PRE_320_u(block)

                if self.block_len == v23c.DG_POST_320_BLOCK_SIZE:
                    self.reserved0 = stream.read(4)

            if self.id != b"DG":
                message = f'Expected "DG" block @{hex(address)} but found "{self.id!r}"'

                logger.exception(message)
                raise MdfException(message)

        except KeyError:
            self.address = kwargs.get("address", 0)
            self.id = b"DG"
            self.block_len = kwargs.get("block_len", v23c.DG_PRE_320_BLOCK_SIZE)
            self.next_dg_addr = kwargs.get("next_dg_addr", 0)
            self.first_cg_addr = kwargs.get("first_cg_addr", 0)
            self.trigger_addr = kwargs.get("comment_addr", 0)
            self.data_block_addr = kwargs.get("data_block_addr", 0)
            self.cg_nr = kwargs.get("cg_nr", 1)
            self.record_id_len = kwargs.get("record_id_len", 0)
            if self.block_len == v23c.DG_POST_320_BLOCK_SIZE:
                self.reserved0 = b"\0\0\0\0"

    def __getitem__(self, item: str) -> object:
        return self.__getattribute__(item)

    def __setitem__(self, item: str, value: object) -> None:
        self.__setattr__(item, value)

    def __bytes__(self) -> bytes:
        keys: tuple[str, ...]
        if self.block_len == v23c.DG_POST_320_BLOCK_SIZE:
            fmt = v23c.FMT_DATA_GROUP_POST_320
            keys = v23c.KEYS_DATA_GROUP_POST_320
        else:
            fmt = v23c.FMT_DATA_GROUP_PRE_320
            keys = v23c.KEYS_DATA_GROUP_PRE_320
        result = pack(fmt, *[self[key] for key in keys])
        return result


class FileIdentificationBlockKwargs(BlockKwargs, total=False):
    version: str


class FileIdentificationBlock:
    """IDBLOCK class.

    IDBLOCK fields:

    * ``file_identification`` -  bytes : file identifier
    * ``version_str`` - bytes : format identifier
    * ``program_identification`` - bytes : creator program identifier
    * ``byte_order`` - int : integer code for byte order (endianness)
    * ``float_format`` - int : integer code for floating-point format
    * ``mdf_version`` - int : version number of MDF format
    * ``code_page`` - int : unicode code page number
    * ``reserved0`` - bytes : reserved bytes
    * ``reserved1`` - bytes : reserved bytes
    * ``unfinalized_standard_flags`` - int : standard flags for unfinalized MDF
    * ``unfinalized_custom_flags`` - int : custom flags for unfinalized MDF

    Other attributes:

    * ``address`` - int : block address inside MDF file; should be 0 always

    Parameters
    ----------
    stream : file handle
        MDF file handle.
    version : int
        MDF version in case of new file (dynamically created).
    """

    __slots__ = (
        "address",
        "byte_order",
        "code_page",
        "file_identification",
        "float_format",
        "mdf_version",
        "program_identification",
        "reserved0",
        "reserved1",
        "unfinalized_custom_flags",
        "unfinalized_standard_flags",
        "version_str",
    )

    def __init__(self, **kwargs: Unpack[FileIdentificationBlockKwargs]) -> None:
        super().__init__()

        self.address = 0
        try:
            stream = kwargs["stream"]
            stream.seek(0)

            (
                self.file_identification,
                self.version_str,
                self.program_identification,
                self.byte_order,
                self.float_format,
                self.mdf_version,
                self.code_page,
                self.reserved0,
                self.reserved1,
                self.unfinalized_standard_flags,
                self.unfinalized_custom_flags,
            ) = typing.cast(v23c.Id, unpack(v23c.ID_FMT, stream.read(v23c.ID_BLOCK_SIZE)))
        except KeyError:
            version = kwargs["version"]
            self.file_identification = "MDF     ".encode("latin-1")
            self.version_str = version.encode("latin-1") + b"\0" * 4
            self.program_identification = f"{tool.__tool_short__}{tool.__version__}".encode("latin-1")
            self.byte_order = v23c.BYTE_ORDER_INTEL if sys.byteorder == "little" else v23c.BYTE_ORDER_MOTOROLA
            self.float_format = 0
            self.mdf_version = int(version.replace(".", ""))
            self.code_page = 0
            self.reserved0 = b"\0" * 2
            self.reserved1 = b"\0" * 26
            self.unfinalized_standard_flags = 0
            self.unfinalized_custom_flags = 0

    def __getitem__(self, item: str) -> object:
        return self.__getattribute__(item)

    def __setitem__(self, item: str, value: object) -> None:
        self.__setattr__(item, value)

    def __bytes__(self) -> bytes:
        result = pack(v23c.ID_FMT, *[self[key] for key in v23c.ID_KEYS])
        return result


class HeaderBlockKwargs(BlockKwargs, total=False):
    version: str


class _CommonProperties(TypedDict, total=False):
    author: str
    project: str
    department: str
    subject: str


class HeaderBlock:
    """HDBLOCK class.

    HDBLOCK fields:

    * ``id`` - bytes : block ID; always b'HD'
    * ``block_len`` - int : block bytes size
    * ``first_dg_addr`` - int : address of first data group block (DGBLOCK)
    * ``comment_addr`` - int : address of TXBLOCK that contains the
      measurement file comment
    * ``program_addr`` - int : address of program block (PRBLOCK)
    * ``dg_nr`` - int : number of data groups
    * ``date`` - bytes : date at which the recording was started in
      "DD:MM:YYYY" format
    * ``time`` - bytes : time at which the recording was started in
      "HH:MM:SS" format
    * ``author_field`` - bytes : author name
    * ``organization_field``` - bytes : organization name
    * ``project_field``` - bytes : project name
    * ``subject_field``` - bytes : subject

    Since version 3.2 the following extra keys were added:

    * ``abs_time`` - int : timestamp at which recording was started in
      nanoseconds.
    * ``tz_offset`` - int : UTC time offset in hours (= GMT time zone)
    * ``time_quality`` - int : time quality class
    * ``timer_identification`` - bytes : timer identification (time source)

    Other attributes:

    * ``address`` - int : block address inside MDF file; should always be 64
    * ``comment`` - int : file comment
    * ``program`` - ProgramBlock : program block
    * ``author`` - str : measurement author
    * ``department`` - str : author department
    * ``project`` - str : working project
    * ``subject`` - str : measurement subject

    Parameters
    ----------
    stream : file handle
        MDF file handle.
    version : int
        MDF version in case of new file (dynamically created).
    """

    def __init__(self, **kwargs: Unpack[HeaderBlockKwargs]) -> None:
        super().__init__()

        self.address = 64
        self.program = None
        self._common_properties: _CommonProperties = {}
        self.description = ""
        self.comment = ""
        try:
            stream = kwargs["stream"]
            stream.seek(64)

            (
                self.id,
                self.block_len,
                self.first_dg_addr,
                self.comment_addr,
                self.program_addr,
                self.dg_nr,
                self.date,
                self.time,
                self.author_field,
                self.department_field,
                self.project_field,
                self.subject_field,
            ) = typing.cast(v23c.HeaderCommon, unpack(v23c.HEADER_COMMON_FMT, stream.read(v23c.HEADER_COMMON_SIZE)))

            if self.block_len > v23c.HEADER_COMMON_SIZE:
                (
                    self.abs_time,
                    self.tz_offset,
                    self.time_quality,
                    self.timer_identification,
                ) = typing.cast(
                    v23c.HeaderPost320Extra,
                    unpack(v23c.HEADER_POST_320_EXTRA_FMT, stream.read(v23c.HEADER_POST_320_EXTRA_SIZE)),
                )

            if self.id != b"HD":
                message = f'Expected "HD" block @{hex(self.address)} but found "{self.id!r}"'
                message = message.format(hex(64), self.id)
                logger.exception(message)
                raise MdfException(message)

            if self.program_addr:
                self.program = ProgramBlock(address=self.program_addr, stream=stream)
            if self.comment_addr:
                self.comment = get_text_v3(address=self.comment_addr, stream=stream)

        except KeyError:
            version = kwargs.get("version", "3.20")
            self.id = b"HD"
            self.block_len = 208 if version >= "3.20" else 164
            self.first_dg_addr = 0
            self.comment_addr = 0
            self.program_addr = 0
            self.dg_nr = 0
            try:
                user = getuser()
            except (ModuleNotFoundError, OSError):
                user = ""
            self.author_field = f"{user:\0<32}".encode("latin-1")
            self.department_field = "{:\0<32}".format("").encode("latin-1")
            self.project_field = "{:\0<32}".format("").encode("latin-1")
            self.subject_field = "{:\0<32}".format("").encode("latin-1")

            if self.block_len > v23c.HEADER_COMMON_SIZE:
                self.time_quality = 0
                self.timer_identification = "{:\0<32}".format("Local PC Reference Time").encode("latin-1")

            self.start_time = datetime(1980, 1, 1, tzinfo=timezone.utc)

        self.author = self.author_field.strip(b" \r\n\t\0").decode("latin-1")
        self.department = self.department_field.strip(b" \r\n\t\0").decode("latin-1")
        self.project = self.project_field.strip(b" \r\n\t\0").decode("latin-1")
        self.subject = self.subject_field.strip(b" \r\n\t\0").decode("latin-1")

    @property
    def comment(self) -> str:
        root = ET.Element("HDcomment")
        text = ET.SubElement(root, "TX")
        text.text = self.description
        common = ET.SubElement(root, "common_properties")
        for name, value in self._common_properties.items():
            if isinstance(value, dict):
                tree = ET.SubElement(common, "tree", name=name)
                for subname, subvalue in value.items():
                    ET.SubElement(tree, "e", name=subname).text = subvalue
            elif isinstance(value, str):
                ET.SubElement(common, "e", name=name).text = value
            else:
                raise TypeError("value must be of type 'dict' or 'str'")

        xml_string: bytes = ET.tostring(root, encoding="utf8", method="xml")

        return xml_string.replace(b"<?xml version='1.0' encoding='utf8'?>\n", b"").decode("utf-8")

    @comment.setter
    def comment(self, string: str) -> None:
        self._common_properties = {}

        if string.startswith("<HDcomment"):
            comment = string
            try:
                comment_xml = ET.fromstring(comment.replace(' xmlns="http://www.asam.net/mdf/v4"', ""))
            except ET.ParseError:
                self.description = string
            else:
                description = comment_xml.find(".//TX")
                if description is None:
                    self.description = ""
                else:
                    self.description = description.text or ""

                common_properties = comment_xml.find(".//common_properties")
                if common_properties is not None:
                    for e in common_properties:
                        if e.tag == "e":
                            name = e.attrib["name"]
                            self._common_properties[name] = e.text or ""  # type: ignore[literal-required]
                        else:
                            name = e.attrib["name"]
                            subattributes: dict[str, str] = {}
                            tree = e
                            self._common_properties[name] = subattributes  # type: ignore[literal-required]

                            for e in tree:
                                name = e.attrib["name"]
                                subattributes[name] = e.text or ""
        else:
            self.description = string

    @property
    def author(self) -> str:
        return self._common_properties.get("author", "")

    @author.setter
    def author(self, value: str) -> None:
        self._common_properties["author"] = value

    @property
    def project(self) -> str:
        return self._common_properties.get("project", "")

    @project.setter
    def project(self, value: str) -> None:
        self._common_properties["project"] = value

    @property
    def department(self) -> str:
        return self._common_properties.get("department", "")

    @department.setter
    def department(self, value: str) -> None:
        self._common_properties["department"] = value

    @property
    def subject(self) -> str:
        return self._common_properties.get("subject", "")

    @subject.setter
    def subject(self, value: str) -> None:
        self._common_properties["subject"] = value

    def to_blocks(
        self,
        address: int,
        blocks: list[SupportsBytes],
        defined_texts: dict[str, int],
        si_map: dict[bytes, int],
    ) -> int:
        blocks.append(self)
        self.address = address
        address += self.block_len

        text = self.comment
        if text:
            if text in defined_texts:
                self.comment_addr = defined_texts[text]
            else:
                tx_block = TextBlock(text=text)
                self.comment_addr = address
                defined_texts[text] = address
                tx_block.address = address
                address += tx_block.block_len
                blocks.append(tx_block)
        else:
            self.comment_addr = 0

        if self.program:
            self.program_addr = address
            address += self.program.block_len
            blocks.append(self.program)

        else:
            self.program_addr = 0

        self.author_field = self.author.encode("latin-1")
        self.department_field = self.department.encode("latin-1")
        self.project_field = self.project.encode("latin-1")
        self.subject_field = self.subject.encode("latin-1")

        return address

    @property
    def start_time(self) -> datetime:
        """Getter and setter of the measurement start timestamp.

        Returns
        -------
        timestamp : datetime.datetime
            Start timestamp.
        """

        try:
            if self.block_len > v23c.HEADER_COMMON_SIZE:
                if self.abs_time:
                    timestamp = self.abs_time / 10**9

                    tz = timezone(timedelta(hours=self.tz_offset))

                    try:
                        timestamp_dt = datetime.fromtimestamp(timestamp, tz)
                    except OverflowError:
                        timestamp_dt = datetime.fromtimestamp(0, tz) + timedelta(seconds=timestamp)
                    except OSError:
                        timestamp_dt = datetime.now(tz)
                else:
                    timestamp_fmt = "{} {}".format(self.date.decode("ascii"), self.time.decode("ascii"))

                    timestamp_dt = datetime.strptime(timestamp_fmt, "%d:%m:%Y %H:%M:%S").astimezone(timezone.utc)

            else:
                timestamp_fmt = "{} {}".format(self.date.decode("ascii"), self.time.decode("ascii"))

                timestamp_dt = datetime.strptime(timestamp_fmt, "%d:%m:%Y %H:%M:%S").astimezone(timezone.utc)
        except:
            return datetime.now().astimezone(timezone.utc)

        return timestamp_dt

    @start_time.setter
    def start_time(self, timestamp: datetime) -> None:
        if timestamp.tzinfo is None:
            # the user provided a naive datetime
            # use the local time zone
            tzinfo = dateutil.tz.tzlocal()
            timestamp = datetime.fromtimestamp(timestamp.timestamp(), tzinfo)

        self.date = timestamp.strftime("%d:%m:%Y").encode("ascii")
        self.time = timestamp.strftime("%H:%M:%S").encode("ascii")
        if self.block_len > v23c.HEADER_COMMON_SIZE:
            if timestamp.tzinfo is None:
                raise ValueError("tzinfo is None")
            offset = timestamp.tzinfo.utcoffset(timestamp)
            if offset is None:
                raise ValueError("utcoffset is None")
            self.tz_offset = int(offset.total_seconds() / 3600)
            self.abs_time = int(timestamp.timestamp() * 10**9)

    def start_time_string(self) -> str:
        tzinfo = self.start_time.tzinfo

        if tzinfo is None:
            raise ValueError("tzinfo is None")

        dst = tzinfo.dst(self.start_time)
        if dst is not None:
            dst = tzinfo.dst(self.start_time)

            if dst is None:
                raise ValueError("dst is None")

            dst_offset = int(dst.total_seconds() / 3600)
        else:
            dst_offset = 0
        offset = tzinfo.utcoffset(self.start_time)

        if offset is None:
            raise ValueError("offset is None")

        tz_offset = int(offset.total_seconds() / 3600) - dst_offset

        tz_offset_sign = "-" if tz_offset < 0 else "+"

        dst_offset_sign = "-" if dst_offset < 0 else "+"

        tz_information = f"[GMT{tz_offset_sign}{tz_offset:.2f} DST{dst_offset_sign}{dst_offset:.2f}h]"

        if self.block_len > v23c.HEADER_COMMON_SIZE:
            if self.abs_time:
                tz_offset_sign = "-" if self.tz_offset < 0 else "+"
                tz_information = f"[GMT{tz_offset_sign}{abs(self.tz_offset):.2f} DST+{0:.2f}h]"

        start_time = f"local time = {self.start_time.strftime('%d-%b-%Y %H:%M:%S + %fu')} {tz_information}"
        return start_time

    def __getitem__(self, item: str) -> object:
        return self.__getattribute__(item)

    def __setitem__(self, item: str, value: object) -> None:
        self.__setattr__(item, value)

    def __bytes__(self) -> bytes:
        fmt = v23c.HEADER_COMMON_FMT
        keys: tuple[str, ...] = v23c.HEADER_COMMON_KEYS
        if self.block_len > v23c.HEADER_COMMON_SIZE:
            fmt += v23c.HEADER_POST_320_EXTRA_FMT
            keys += v23c.HEADER_POST_320_EXTRA_KEYS
        result = pack(fmt, *[self[key] for key in keys])
        return result


class ProgramBlockKwargs(BlockKwargs, total=False):
    data: bytes


class ProgramBlock:
    """PRBLOCK class.

    PRBLOCK fields:

    * ``id`` - bytes : block ID; always b'PR'
    * ``block_len`` - int : block bytes size
    * ``data`` - bytes : creator program free format data

    Other attributes:

    * ``address`` - int : block address inside MDF file

    Parameters
    ----------
    stream : file handle
        MDF file handle.
    address : int
        Block address inside MDF file.
    """

    __slots__ = ("address", "block_len", "data", "id")

    def __init__(self, **kwargs: Unpack[ProgramBlockKwargs]) -> None:
        super().__init__()

        try:
            stream = kwargs["stream"]
            self.address = address = kwargs["address"]
            stream.seek(address)

            (self.id, self.block_len) = COMMON_u(stream.read(4))
            self.data = stream.read(self.block_len - 4)

            if self.id != b"PR":
                message = f'Expected "PR" block @{hex(address)} but found "{self.id!r}"'

                logger.exception(message)
                raise MdfException(message)

        except KeyError:
            self.id = b"PR"
            self.block_len = len(kwargs["data"]) + 6
            self.data = kwargs["data"]

    def __getitem__(self, item: str) -> object:
        return self.__getattribute__(item)

    def __setitem__(self, item: str, value: object) -> None:
        self.__setattr__(item, value)

    def __bytes__(self) -> bytes:
        fmt = v23c.FMT_PROGRAM_BLOCK.format(self.block_len - 4)
        result = pack(fmt, *[self[key] for key in v23c.KEYS_PROGRAM_BLOCK])
        return result


class TextBlockKwargs(BlockKwargs, total=False):
    text: bytes | str


class TextBlock:
    r"""TXBLOCK class.

    TXBLOCK fields:

    * ``id`` - bytes : block ID; always b'TX'
    * ``block_len`` - int : block bytes size
    * ``text`` - bytes : text content

    Other attributes:

    * ``address`` - int : block address inside MDF file

    Parameters
    ----------
    stream : file handle
        MDF file handle.
    address : int
        Block address inside MDF file.
    text : bytes | str
        Bytes or str for creating a new TextBlock.

    Examples
    --------
    >>> tx1 = TextBlock(text='VehicleSpeed')
    >>> tx1['text']
    b'VehicleSpeed\x00'
    """

    __slots__ = ("address", "block_len", "id", "text")

    def __init__(self, **kwargs: Unpack[TextBlockKwargs]) -> None:
        super().__init__()
        try:
            stream = kwargs["stream"]
            mapped = kwargs.get("mapped", False)
            self.address = address = kwargs["address"]
            if utils.stream_is_mmap(stream, mapped):
                (self.id, self.block_len) = COMMON_uf(stream, address)
                if self.id != b"TX":
                    message = f'Expected "TX" block @{hex(address)} but found "{self.id!r}"'
                    logger.exception(message)
                    raise MdfException(message)

                self.text = stream[address + 4 : address + self.block_len]
            else:
                stream.seek(address)
                (self.id, self.block_len) = COMMON_u(stream.read(4))
                if self.id != b"TX":
                    message = f'Expected "TX" block @{hex(address)} but found "{self.id!r}"'
                    logger.exception(message)
                    raise MdfException(message)

                size = self.block_len - 4
                self.text = stream.read(size)

        except KeyError:
            self.address = 0
            text = kwargs["text"]

            if isinstance(text, str):
                text_bytes = text.encode("latin-1", "backslashreplace")
            else:
                text_bytes = text

            self.id = b"TX"
            self.block_len = len(text_bytes) + 5
            self.text = text_bytes + b"\0"

            if self.block_len > 65000:
                self.block_len = 65000 + 5
                self.text = self.text[:65000] + b"\0"

    def __getitem__(self, item: str) -> object:
        return self.__getattribute__(item)

    def __setitem__(self, item: str, value: object) -> None:
        self.__setattr__(item, value)

    def __bytes__(self) -> bytes:
        return v23c.COMMON_p(self.id, self.block_len) + self.text

    def __repr__(self) -> str:
        return f"TextBlock(id={self.id!r}, block_len={self.block_len}, text={self.text!r})"


class TriggerBlock:
    """TRBLOCK class.

    TRBLOCK fields:

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

    Other attributes:

    * ``address`` - int : block address inside MDF file
    * ``comment`` - str : trigger comment

    Parameters
    ----------
    stream : file handle
        MDF file handle.
    address : int
        Block address inside MDF file.
    """

    def __init__(self, **kwargs: Unpack[BlockKwargs]) -> None:
        super().__init__()

        self.comment = ""

        try:
            self.address = address = kwargs["address"]
            stream = kwargs["stream"]

            stream.seek(address + 2)
            (size,) = UINT16_u(stream.read(2))
            stream.seek(address)
            block = stream.read(size)

            (self.id, self.block_len, self.text_addr, self.trigger_events_nr) = typing.cast(
                tuple[bytes, int, int, int], unpack("<2sHIH", block[:10])
            )

            nr = self.trigger_events_nr
            if nr:
                values: tuple[float, ...] = unpack(f"<{3 * nr}d", block[10:])
            for i in range(nr):
                (
                    self[f"trigger_{i}_time"],
                    self[f"trigger_{i}_pretime"],
                    self[f"trigger_{i}_posttime"],
                ) = (values[i * 3], values[3 * i + 1], values[3 * i + 2])

            if self.text_addr:
                self.comment = get_text_v3(address=self.text_addr, stream=stream)

            if self.id != b"TR":
                message = f'Expected "TR" block @{hex(address)} but found "{self.id!r}"'

                logger.exception(message)
                raise MdfException(message)

        except KeyError:
            self.address = 0
            nr = 0
            while f"trigger_{nr}_time" in kwargs:
                nr += 1

            self.id = b"TR"
            self.block_len = 10 + nr * 8 * 3
            self.text_addr = 0
            self.trigger_events_nr = nr

            for i in range(nr):
                key = f"trigger_{i}_time"
                self[key] = kwargs[key]  # type: ignore[literal-required]
                key = f"trigger_{i}_pretime"
                self[key] = kwargs[key]  # type: ignore[literal-required]
                key = f"trigger_{i}_posttime"
                self[key] = kwargs[key]  # type: ignore[literal-required]

    def to_blocks(self, address: int, blocks: list[bytes | SupportsBytes]) -> int:
        text = self.comment
        if text:
            tx_block = TextBlock(text=text)
            self.text_addr = address
            address += tx_block.block_len
            blocks.append(tx_block)
        else:
            self.text_addr = 0

        blocks.append(self)
        self.address = address
        address += self.block_len

        return address

    def __getitem__(self, item: str) -> object:
        return self.__getattribute__(item)

    def __setitem__(self, item: str, value: object) -> None:
        self.__setattr__(item, value)

    def __bytes__(self) -> bytes:
        triggers_nr = self.trigger_events_nr
        fmt = f"<2sHIH{triggers_nr * 3}d"
        keys: tuple[str, ...] = ("id", "block_len", "text_addr", "trigger_events_nr")
        for i in range(triggers_nr):
            keys += (
                f"trigger_{i}_time",
                f"trigger_{i}_pretime",
                f"trigger_{i}_posttime",
            )
        result = pack(fmt, *[self[key] for key in keys])
        return result

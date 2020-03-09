import os
import struct

from enum import IntFlag
from typing import Optional, Tuple, BinaryIO, Union, Dict

from asammdf.blocks.utils import MdfException


class FinalizationFlags(IntFlag):
    CG_CA_CYCLE_COUNTERS = 1 << 0
    SR_CYCLE_COUNTERS = 1 << 1
    DT_LENGTH = 1 << 2
    RD_LENGTH = 1 << 3
    UPDATE_DL = 1 << 4
    VLSD_CG_DATA_BYTES = 1 << 5
    VLSD_CG_OFFSET = 1 << 6


class MdfHeader:
    """Representation of the MDF header.
    """

    _fmt = "<4s4xQQ"
    _fmt_length = 24

    def __init__(
        self, identification_str: str = None, length: int = 0, link_count: int = 0
    ):
        self.identification_str = identification_str
        self.length = length
        self.link_count = link_count

    def __bytes__(self):
        return struct.pack(
            self._fmt, self.identification_str, self.length, self.link_count
        )

    def __len__(self):
        return self._fmt_length

    pass


class MdfBlock:
    """Generic representation of a MDF4 block (Except the ID block).
    """

    def __init__(self, header: MdfHeader):
        self.header: MdfHeader = header
        self.file_location: int = 0
        self.links: Tuple[int, ...] = (0,) * header.link_count

        return

    def load(self, stream: BinaryIO):
        """Function responsible for loading anything past the header of the block.

        Expects the stream position to be just after the header.

        Parameters
        ----------
        stream : BinaryIO
            Access to block data
        """
        # Load the addresses of the block links.
        self.links = struct.unpack(
            f"<{self.header.link_count}Q", stream.read(8 * self.header.link_count)
        )

        return

    def __bytes__(self):
        # Aggregate the result of the header and the links.
        result = bytes(self.header)
        result += struct.pack(f"<{self.header.link_count}Q", *self.links)

        return result

    pass


class CG_Block(MdfBlock):
    _fmt = "<QQHH4xLL"
    _fmt_length = 32

    def __init__(self, header: MdfHeader = None):
        if not header:
            header = MdfHeader(b"##CG", 104, 6)
        super().__init__(header)

        # Data fields.
        self.record_id = 0
        self.cycle_count = 0
        self.flags = 0
        self.path_separator = 0
        self.data_bytes = 0
        self.inval_bytes = 0

        return

    @property
    def is_vlsd(self) -> bool:
        """Is this a VLSD block.

        Returns
        -------
        bool
            True if VLSD block, False otherwise.
        """
        return (self.flags & (1 << 0)) == 1

    @property
    def get_next_cg_block_address(self) -> int:
        """Get the next CG block address in this chain of CG blocks.

        Returns
        -------
        int
            The address of the next CG block, or 0 if none exists.
        """
        return self.links[0]

    @property
    def get_first_cn_block_address(self) -> int:
        """Get the first CN block address for this CG block.

        Returns
        -------
        int
            The address of the first CN block, or 0 if none exists.
        """
        return self.links[1]

    def load(self, stream: BinaryIO):
        # Load links via super class.
        super().load(stream)

        # Load data.
        (
            self.record_id,
            self.cycle_count,
            self.flags,
            self.path_separator,
            self.data_bytes,
            self.inval_bytes,
        ) = struct.unpack(self._fmt, stream.read(self._fmt_length))

        return

    def __bytes__(self):
        # Aggregate the result of the header and links from the super class, and the data from this class.
        result = super().__bytes__()

        result += struct.pack(
            self._fmt,
            self.record_id,
            self.cycle_count,
            self.flags,
            self.path_separator,
            self.data_bytes,
            self.inval_bytes,
        )

        return result

    pass


class CN_Block(MdfBlock):
    _fmt = "<4B4LBxH6d"
    _fmt_length = 72

    def __init__(self, header: MdfHeader = None):
        if not header:
            header = MdfHeader(b"##CN", 8, 160)
        super().__init__(header)

        # Data fields.
        self.channel_type = 0
        self.sync_type = 0
        self.data_type = 0
        self.bit_offset = 0
        self.byte_offset = 0
        self.bit_count = 0
        self.flags = 0
        self.pos_invalidation_bit = 0
        self.precision = 0
        self.attachment_nr = 0
        self.min_raw_value = 0
        self.max_raw_value = 0
        self.lower_limit = 0
        self.upper_limit = 0
        self.lower_ext_limit = 0
        self.upper_ext_limit = 0

        return

    @property
    def get_next_cn_block_address(self):
        """Get the next CN block address in this chain of CN blocks.

        Returns
        -------
        int
            The address of the next CN block, or 0 if none exists.
        """
        return self.links[0]

    def load(self, stream: BinaryIO):
        # Load links via super class.
        super().load(stream)

        # Load data.
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
            self.attachment_nr,
            self.min_raw_value,
            self.max_raw_value,
            self.lower_limit,
            self.upper_limit,
            self.lower_ext_limit,
            self.upper_ext_limit,
        ) = struct.unpack(self._fmt, stream.read(self._fmt_length))

        return

    def __bytes__(self):
        # Aggregate the result of the header and links from the super class, and the data from this class.
        result = super().__bytes__()

        result += struct.pack(
            self._fmt,
            self.channel_type,
            self.sync_type,
            self.data_type,
            self.bit_offset,
            self.byte_offset,
            self.bit_count,
            self.flags,
            self.pos_invalidation_bit,
            self.precision,
            self.attachment_nr,
            self.min_raw_value,
            self.max_raw_value,
            self.lower_limit,
            self.upper_limit,
            self.lower_ext_limit,
            self.upper_ext_limit,
        )

        return result

    pass


class DG_Block(MdfBlock):
    _fmt = "<B7x"
    _fmt_length = 8

    def __init__(self, header: MdfHeader = None):
        if not header:
            header = MdfHeader(b"##DG", 64, 4)
        super().__init__(header)

        # Data fields.
        self.record_size = 0

        return

    @property
    def get_next_dg_block_address(self):
        """Get the next DG block address in this chain of DG blocks.

        Returns
        -------
        int
            The address of the next DG block, or 0 if none exists.
        """
        return self.links[0]

    @property
    def get_cg_block_address(self):
        """Get the address of the linked CG block for this DG block.

        Returns
        -------
        int
            The address of the linked CG block, or 0 if none exists.
        """
        return self.links[1]

    @property
    def get_dt_block_address(self):
        """Get the address of the linked DT block for this DG block.

        Returns
        -------
        int
            The address of the linked DT block, or 0 if none exists.
        """
        return self.links[2]

    def load(self, stream: BinaryIO):
        # Load links via super class.
        super().load(stream)

        # Load data.
        (self.record_size,) = struct.unpack(self._fmt, stream.read(self._fmt_length))
        pass

    def __bytes__(self):
        # Aggregate the result of the header and links from the super class, and the data from this class.
        result = super().__bytes__()

        result += struct.pack(self._fmt, self.record_size,)

        return result

    pass


class HD_Block(MdfBlock):
    _fmt = "<Q2h3Bx2d"
    _fmt_length = 32

    def __init__(self, header: MdfHeader = None):
        if not header:
            header = MdfHeader(b"##HD", 104, 6)
        super().__init__(header)

        # Data fields.
        self.abs_time = 0
        self.tz_offset = 0
        self.daylight_save_time = 0
        self.time_flags = 0
        self.time_quality = 0
        self.flags = 0
        self.start_angle = 0
        self.start_distance = 0

        return

    @property
    def get_dg_block_address(self):
        """Get the address of the linked DG block for this HD block.

        Returns
        -------
        int
            The address of the linked DG block, or 0 if none exists.
        """
        return self.links[1]

    def load(self, stream: BinaryIO):
        # Load links via super class.
        super().load(stream)

        # Load data.
        (
            self.abs_time,
            self.tz_offset,
            self.daylight_save_time,
            self.time_flags,
            self.time_quality,
            self.flags,
            self.start_angle,
            self.start_distance,
        ) = struct.unpack(self._fmt, stream.read(self._fmt_length))
        pass

    def __bytes__(self):
        # Aggregate the result of the header and links from the super class, and the data from this class.
        result = super().__bytes__()

        result += struct.pack(
            self._fmt,
            self.abs_time,
            self.tz_offset,
            self.daylight_save_time,
            self.time_flags,
            self.time_quality,
            self.flags,
            self.start_angle,
            self.start_distance,
        )

        return result

    pass


class FinalizationShim:
    """Shim class to insert between un-finalized MDF files and asammdf.
    """

    def __init__(self, parent: BinaryIO, finalization_flags: int):
        super().__init__()

        # Check that the requested flags are supported.
        supported_flags = FinalizationFlags.VLSD_CG_DATA_BYTES
        supported_flags |= FinalizationFlags.CG_CA_CYCLE_COUNTERS
        supported_flags |= FinalizationFlags.DT_LENGTH

        if finalization_flags & ~supported_flags:
            raise MdfException("Unsupported finalization options detected")

        self._parent = parent
        self._blocks: {int, MdfBlock} = {}
        self._flags = finalization_flags

        # Determine the file size by seeking to the end of the stream.
        self._parent.seek(0, os.SEEK_END)
        self._file_size = self._parent.tell()

        # Keep track of the local/shimmed position.
        self._location = 0

        return

    def load_blocks(self):
        """Creates a tree of all the blocks.
        """

        # Load all blocks recursively, starting with the HD block at address 64.
        self._load_block(64)

    def finalize(self):
        """Attempts to perform in-memory finalization.
        """
        # Look at the finalization flags.
        if self._flags & FinalizationFlags.DT_LENGTH:
            self._handle_dt()
            self._flags = ~FinalizationFlags.DT_LENGTH & self._flags
            pass
        if (self._flags & FinalizationFlags.CG_CA_CYCLE_COUNTERS) or (
            self._flags & FinalizationFlags.VLSD_CG_DATA_BYTES
        ):
            # Since all data records have to be read to get valid cycle counts, data bytes can be updated for "free".
            self._handle_cg_counters()
            self._flags = ~FinalizationFlags.CG_CA_CYCLE_COUNTERS & self._flags
            self._flags = ~FinalizationFlags.VLSD_CG_DATA_BYTES & self._flags
            pass

        return

    # Overload methods of BinaryIO.
    def close(self):
        # Delegate to enclosed stream.
        self._parent.close()
        return

    def read(self, size: Optional[int] = ...) -> bytes:
        if size == ...:
            size = 0

        # Does the shim contain a block in the requested memory area?
        match: Optional[MdfBlock] = None

        for block in self._blocks.values():
            start = block.file_location
            end = start + block.header.length

            if start <= self._location < end:
                match = block
                break

        result = None

        if match is not None:
            # Replace returned data with data from the loaded block in the shim.
            data = bytes(match)
            offset = self._location - match.file_location

            if len(data) - offset > size:
                result = data[offset : offset + size]
                self._location += size
                size = 0
            else:
                result = data[offset:]
                self._location += len(result)
                size -= len(result)

        # Supply with additional data from the original file if necessary.
        self._parent.seek(self._location)
        additional_data = self._parent.read(size)
        self._location += size

        result += additional_data

        return result

    def seek(self, offset: int, whence: int = ...) -> int:
        if whence == ...:
            whence = 0

        if whence == 0:
            self._location = offset
        elif whence == 2:
            self._location = self._file_size - offset
        return self._parent.seek(offset, whence)

    def tell(self):
        return self._parent.tell()

    # Python overrides.
    def __iter__(self):
        return self

    def __next__(self):
        self._parent.seek(self._location)
        result = self._parent.read(1)
        self._location += 1

        return result

    def _handle_dt(self):
        """Update the length of the last DT block. Must be bounded by end of file (EOF) or another block.
        """

        dt_block: Optional[MdfBlock] = None

        # Start by iterating over the loaded blocks in reverse, looking for a DT block.
        for key in reversed(sorted(self._blocks.keys())):
            block = self._blocks[key]

            if block.header.identification_str == b"##DT":
                dt_block = block
                break

        if dt_block is None:
            raise RuntimeError(
                "Requested to finalize DT block, but no DT blocks found in the file."
            )

        # Set the initial guess at a bound to the EOF.
        bound = self._file_size

        # Determine if bound by another block.
        keys = sorted(self._blocks.keys())

        for i, key in enumerate(keys):
            if key == dt_block.file_location:
                # Check if a block exists after this.
                if len(self._blocks) - 1 != i:
                    bound = keys[i + 1]
                    break

        # Update block size.
        dt_block.header.length = bound - dt_block.file_location

        return

    def _handle_cg_counters(self):
        """Update CG counters, set data bytes for VLSD CG blocks.
        """

        # Find all DG blocks and their corresponding data block. Loop over all CG blocks in each DG block.
        hd_block = self._blocks[64]
        dg_block: DG_Block = self._blocks.get(hd_block.links[0])

        while dg_block is not None:
            # Find all CG blocks for this DG block.
            cg_blocks: [CG_Block] = []
            cg_block: CG_Block = self._blocks.get(dg_block.get_cg_block_address)

            while cg_block is not None:
                cg_blocks.append(cg_block)
                cg_block = self._blocks.get(cg_block.get_next_cg_block_address)
                pass

            # Determine the number of cycles in each, as well as number of bytes in CG VLSD blocks. This is done by
            # looping over the records in the DT block, so the record size as well as record IDs along with the sizes
            # of the static data can to be determined.
            dg_record_size = dg_block.record_size
            dg_record_fmt = ""
            if dg_record_size == 1:
                dg_record_fmt = "<B"
            elif dg_record_size == 2:
                dg_record_fmt = "<H"
            elif dg_record_size == 4:
                dg_record_fmt = "<L"
            elif dg_record_size == 8:
                dg_record_fmt = "<Q"

            size_map = {}

            for cg_block in cg_blocks:
                cg_block_id = cg_block.record_id

                if cg_block.is_vlsd:
                    # Invalid size to signify this is a VLSD record, which should obtain the size using another method.
                    cg_block_size = -1
                else:
                    # Count the number of bits used by all sub-channels. This is done by taking the highest bit/byte
                    # offset, and utilising the bit count there.
                    cn_block: CN_Block = self._blocks.get(
                        cg_block.get_first_cn_block_address
                    )
                    highest_bit_value = 0

                    while cn_block is not None:
                        bit_value = (
                            cn_block.byte_offset * 8
                            + cn_block.bit_offset
                            + cn_block.bit_count
                        )

                        if bit_value > highest_bit_value:
                            highest_bit_value = bit_value

                        cn_block = self._blocks.get(cn_block.get_next_cn_block_address)
                        pass

                    # Convert from bits to a byte value.
                    cg_block_size = highest_bit_value // 8

                    remainder = highest_bit_value % 8
                    if remainder != 0:
                        cg_block_size += 1
                    pass

                # Store the result as a new record in the size dictionary.
                size_map[cg_block_id] = cg_block_size

            # Loop over the records in the DT block, and extract record counts and data lengths.
            dt_location = 0
            dt_block = self._blocks.get(dg_block.get_dt_block_address)
            dt_block_size = dt_block.header.length - len(dt_block.header)

            self._parent.seek(dt_block.file_location + len(dt_block.header))

            # Prepare storage for cycle counters and data sizes.
            cycle_counters: Dict[int, int] = {}
            data_bytes: Dict[int, int] = {}
            for key in size_map.keys():
                cycle_counters[key] = 0
                data_bytes[key] = 0
                pass

            while dt_location < dt_block_size:
                # Read record id.
                (record_id,) = struct.unpack(
                    dg_record_fmt, self._parent.read(dg_record_size)
                )
                dt_location += dg_record_size

                # Increment the corresponding cycle counter.
                cycle_counters[record_id] += 1

                # Determine if this is static or dynamic data.
                record_length = size_map.get(record_id)

                if record_length < 0:
                    # VLSD data, read 4 bytes of length information from the data record.
                    (record_length,) = struct.unpack("<L", self._parent.read(4))
                    dt_location += 4
                    pass

                # Skip over the data, but store the size.
                self._parent.read(record_length)
                data_bytes[record_id] += record_length
                dt_location += record_length
                pass

            # Update the cycle counters and data bytes if requested.
            for cg_block in cg_blocks:
                record_id = cg_block.record_id

                if self._flags & FinalizationFlags.CG_CA_CYCLE_COUNTERS:
                    cg_block.cycle_count = cycle_counters.get(record_id)

                if cg_block.is_vlsd and (
                    self._flags & FinalizationFlags.VLSD_CG_DATA_BYTES
                ):
                    cg_block.data_bytes = data_bytes.get(record_id) & 0x00000000FFFFFFFF
                    cg_block.inval_bytes = (
                        data_bytes.get(record_id) & 0xFFFFFFFF00000000
                    ) << 32

            dg_block = self._blocks.get(dg_block.get_next_dg_block_address)
            pass

        return

    def _load_block(self, address: int):
        """Load a MDF block and all linked blocks.

        Parameters
        ----------
        address : int
            The file address to load the block at.

        Returns
        -------
        MDfBlock
            The loaded block.
        """
        # Read the header bytes.
        self._parent.seek(address)

        data = struct.unpack("<4s4x2Q", self._parent.read(24))
        header = MdfHeader(*data)

        # Determine the block type.
        block = None
        if header.identification_str == b"##CG":
            block = CG_Block(header)
        elif header.identification_str == b"##CN":
            block = CN_Block(header)
        elif header.identification_str == b"##DG":
            block = DG_Block(header)
        elif header.identification_str == b"##HD":
            block = HD_Block(header)
        else:
            block = MdfBlock(header)

        block.file_location = address
        block.load(self._parent)

        self._blocks[address] = block

        for link_address in block.links:
            if link_address == 0:
                continue

            self._load_block(link_address)
            pass

        return block

    pass

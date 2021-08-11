"""
ASAM MDF version 4 file format module
"""

import bisect
from collections import defaultdict
from functools import lru_cache
from hashlib import md5
import logging
from math import ceil
import mmap
import os
from pathlib import Path
import shutil
import sys
from tempfile import gettempdir, TemporaryFile
from traceback import format_exc
from zipfile import ZIP_DEFLATED, ZipFile
from zlib import decompress

import canmatrix
from lz4.frame import compress as lz_compress
from lz4.frame import decompress as lz_decompress
from numpy import (
    arange,
    argwhere,
    array,
    array_equal,
    column_stack,
    concatenate,
    cumsum,
    dtype,
    empty,
    fliplr,
    float32,
    float64,
    frombuffer,
    full,
    linspace,
    nonzero,
    packbits,
    searchsorted,
    transpose,
    uint8,
    uint16,
    uint32,
    uint64,
    unique,
    where,
    zeros,
)
from numpy.core.defchararray import decode, encode
from numpy.core.records import fromarrays, fromstring
from pandas import DataFrame

from . import encryption
from . import v4_constants as v4c
from ..signal import Signal
from ..version import __version__
from .bus_logging_utils import extract_mux
from .conversion_utils import conversion_transfer
from .mdf_common import MDF_Common
from .source_utils import Source
from .utils import (
    all_blocks_addresses,
    as_non_byte_sized_signed_int,
    CHANNEL_COUNT,
    ChannelsDB,
    CONVERT,
    count_channel_groups,
    DataBlockInfo,
    debug_channel,
    extract_cncomment_xml,
    extract_display_name,
    fmt_to_datatype_v4,
    get_fmt_v4,
    get_text_v4,
    Group,
    InvalidationBlockInfo,
    is_file_like,
    load_can_database,
    MdfException,
    sanitize_xml,
    SignalDataBlockInfo,
    UINT8_uf,
    UINT16_uf,
    UINT32_p,
    UINT32_uf,
    UINT64_uf,
    UniqueDB,
    validate_version_argument,
    VirtualChannelGroup,
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
    ListData,
    SourceInformation,
    TextBlock,
)

MASTER_CHANNELS = (v4c.CHANNEL_TYPE_MASTER, v4c.CHANNEL_TYPE_VIRTUAL_MASTER)
COMMON_SIZE = v4c.COMMON_SIZE
COMMON_u = v4c.COMMON_u
COMMON_uf = v4c.COMMON_uf

COMMON_SHORT_SIZE = v4c.COMMON_SHORT_SIZE
COMMON_SHORT_uf = v4c.COMMON_SHORT_uf
COMMON_SHORT_u = v4c.COMMON_SHORT_u


logger = logging.getLogger("asammdf")

__all__ = ["MDF4"]


try:
    from .cutils import extract, get_vlsd_offsets, lengths, sort_data_block

#    for now avoid usign the cextension code
#    2/0
except:

    def extract(signal_data, is_byte_array, offsets=()):
        #        offsets_ = set(offsets)
        size = len(signal_data)
        positions = []
        values = []
        pos = 0

        while pos < size:

            positions.append(pos)
            #            if offsets_ and pos not in offsets_:
            #                raise Exception(f"VLSD offsets do not match the signal data:\n{positions}\n{offsets[:len(positions)]}")
            (str_size,) = UINT32_uf(signal_data, pos)
            pos = pos + 4 + str_size
            values.append(signal_data[pos - str_size : pos])

        if is_byte_array:

            values = array(values)
            values = values.view(dtype=f"({values.itemsize},)u1")
        else:

            values = array(values)

        return values

    def sort_data_block(
        signal_data, partial_records, cg_size, record_id_nr, _unpack_stuct
    ):
        """Reads an unsorted DTBLOCK and writes the results to `partial_records`.

        Args:
            signal_data (bytes): DTBLOCK contents
            partial_records (dict): dictionary with `cg_record_id` as key and list of bytes
                as value.
            cg_size (dict): Dictionary with `cg_record_id` as key and
                number of record databytes (i.e. `cg_data_bytes`)
            record_id_nr (int): Number of Bytes used for record IDs
                in the data block (`dg_rec_id_size`).
            _unpack_stuct (callable): Struct("...").unpack_from callable

        Returns:
            bytes: rest of data which couldn't be parsed, can be used in consecutive
                reading attempt
        """
        i = 0
        size = len(signal_data)
        pos = 0
        rem = b""
        while i + record_id_nr < size:
            (rec_id,) = _unpack_stuct(signal_data, i)
            # skip record id
            i += record_id_nr
            try:
                rec_size = cg_size[rec_id]
            except:
                return b""

            if rec_size:
                if rec_size + i > size:
                    rem = signal_data[pos:]
                    break
                endpoint = i + rec_size
                partial_records[rec_id].append(signal_data[i:endpoint])
                i = endpoint
            else:
                if i + 4 > size:
                    rem = signal_data[pos:]
                    break
                (rec_size,) = UINT32_uf(signal_data, i)
                endpoint = i + rec_size + 4
                if endpoint > size:
                    rem = signal_data[pos:]
                    break
                partial_records[rec_id].append(signal_data[i:endpoint])
                i = endpoint
            pos = i
        else:
            rem = signal_data[pos:]

        return rem

    def lengths(iterable):
        return [len(item) for item in iterable]

    def get_vlsd_offsets(data):
        offsets = [0] + [len(item) for item in data]
        offsets = cumsum(offsets)
        return offsets[:-1], offsets[-1]


class MDF4(MDF_Common):
    """The *header* attibute is a *HeaderBlock*.

    The *groups* attribute is a list of dicts, each one with the following keys:

    * ``data_group`` - DataGroup object
    * ``channel_group`` - ChannelGroup object
    * ``channels`` - list of Channel objects with the same order as found in the mdf file
    * ``channel_dependencies`` - list of *ChannelArrayBlock* in case of channel arrays;
      list of Channel objects in case of structure channel composition
    * ``data_block`` - address of data block
    * ``data_location``- integer code for data location (original file, temporary file or
      memory)
    * ``data_block_addr`` - list of raw samples starting addresses
    * ``data_block_type`` - list of codes for data block type
    * ``data_block_size`` - list of raw samples block size
    * ``sorted`` - sorted indicator flag
    * ``record_size`` - dict that maps record ID's to record sizes in bytes (including invalidation bytes)
    * ``param`` - row size used for transposition, in case of transposed zipped blocks


    Parameters
    ----------
    name : string
        mdf file name (if provided it must be a real file name) or
        file-like object

    version : string
        mdf file version ('4.00', '4.10', '4.11', '4.20'); default '4.10'

    kwargs :

    callback : function
        keyword only argument: function to call to update the progress; the
        function must accept two arguments (the current progress and maximum
        progress value)
    use_display_names : bool
        keyword only argument: for MDF4 files parse the XML channel comment to
        search for the display name; XML parsing is quite expensive so setting
        this to *False* can decrease the loading times very much; default
        *False*
    remove_source_from_channel_names (True) : bool

    copy_on_get (True) : bool
        copy channel values (np.array) to avoid high memory usage
    compact_vlsd (False) : bool
        use slower method to save the exact sample size for VLSD channels
    column_storage (True) : bool
        use column storage for MDF version >= 4.20
    encryption_key : bytes
        use this key to decode encrypted attachments

    Attributes
    ----------
    attachments : list
        list of file attachments
    channels_db : dict
        used for fast channel access by name; for each name key the value is a
        list of (group index, channel index) tuples
    events : list
        list event blocks
    file_comment : TextBlock
        file comment TextBlock
    file_history : list
        list of (FileHistory, TextBlock) pairs
    groups : list
        list of data group dicts
    header : HeaderBlock
        mdf file header
    identification : FileIdentificationBlock
        mdf file start block
    last_call_info : dict | None
        a dict to hold information about the last called method.

        .. versionadded:: 5.12.0

    masters_db : dict
        used for fast master channel access; for each group index key the value
         is the master channel index
    name : string
        mdf file name
    version : str
        mdf version

    """

    _terminate = False

    def __init__(self, name=None, version="4.10", channels=None, **kwargs):

        self._kwargs = kwargs
        self.original_name = kwargs["original_name"]
        self.groups = []
        self.header = None
        self.identification = None
        self.file_history = []
        self.channels_db = ChannelsDB()
        self.masters_db = {}
        self.attachments = []
        self._attachments_cache = {}
        self.file_comment = None
        self.events = []
        self.bus_logging_map = {"CAN": {}, "ETHERNET": {}, "FLEXRAY": {}, "LIN": {}}

        self._attachments_map = {}
        self._ch_map = {}
        self._master_channel_metadata = {}
        self._invalidation_cache = {}
        self._external_dbc_cache = {}
        self._si_map = {}
        self._file_si_map = {}
        self._cc_map = {}
        self._file_cc_map = {}
        self._cg_map = {}
        self._cn_data_map = {}
        self._dbc_cache = {}
        self._interned_strings = {}

        if channels is None:
            self.load_filter = set()
            self.use_load_filter = False
        else:
            self.load_filter = set(channels)
            self.use_load_filter = True

        self._tempfile = TemporaryFile()
        self._file = None

        self._raise_on_multiple_occurrences = True
        self._read_fragment_size = 0 * 2 ** 20
        self._write_fragment_size = 4 * 2 ** 20
        self._use_display_names = kwargs.get("use_display_names", False)
        self._remove_source_from_channel_names = kwargs.get(
            "remove_source_from_channel_names", False
        )
        self._encryption_function = kwargs.get("encryption_function", None)
        self._decryption_function = kwargs.get("decryption_function", None)
        self.copy_on_get = kwargs.get("copy_on_get", True)
        self.compact_vlsd = kwargs.get("compact_vlsd", False)
        self._single_bit_uint_as_bool = False
        self._integer_interpolation = 0
        self._float_interpolation = 1
        self.virtual_groups = {}  # master group 2 referencing groups
        self.virtual_groups_map = {}  # group index 2 master group

        self._master = None

        self.last_call_info = None

        # make sure no appended block has the address 0
        self._tempfile.write(b"\0")

        self._callback = kwargs.get("callback", None)

        self._delete_on_close = False

        if name:
            if is_file_like(name):
                self._file = name
                self.name = self.original_name = Path("From_FileLike.mf4")
                self._from_filelike = True
                self._read(mapped=False)
            else:

                with open(name, "rb") as stream:
                    identification = FileIdentificationBlock(stream=stream)
                    version = identification["version_str"]
                    version = version.decode("utf-8").strip(" \n\t\0")
                    flags = identification["unfinalized_standard_flags"]

                if version >= "4.10" and flags:
                    tmpdir = Path(gettempdir())
                    self.name = tmpdir / Path(name).name
                    shutil.copy(name, self.name)
                    self._file = open(self.name, "rb+")
                    self._from_filelike = False
                    self._delete_on_close = True
                    self._read(mapped=False)
                else:

                    if sys.maxsize < 2 ** 32:
                        self.name = Path(name)
                        self._file = open(self.name, "rb")
                        self._from_filelike = False
                        self._read(mapped=False)
                    else:
                        self.name = Path(name)
                        x = open(self.name, "rb")
                        self._file = mmap.mmap(x.fileno(), 0, access=mmap.ACCESS_READ)
                        self._from_filelike = False
                        self._read(mapped=True)

                        self._file.close()
                        x.close()

                        self._file = open(self.name, "rb")

        else:
            self._from_filelike = False
            version = validate_version_argument(version)
            self.header = HeaderBlock()
            self.identification = FileIdentificationBlock(version=version)
            self.version = version
            self.name = Path("__new__.mf4")

        if self.version >= "4.20":
            self._column_storage = kwargs.get("column_storage", True)
        else:
            self._column_storage = False

        self._parent = None

    def __del__(self):
        self.close()

    def _check_finalised(self) -> bool:
        flags = self.identification["unfinalized_standard_flags"]

        if flags & 1:
            message = (
                f"Unfinalised file {self.name}:"
                " Update of cycle counters for CG/CA blocks required"
            )

            logger.info(message)
        if flags & 1 << 1:
            message = f"Unfinalised file {self.name}: Update of cycle counters for SR blocks required"

            logger.info(message)
        if flags & 1 << 2:
            message = f"Unfinalised file {self.name}: Update of length for last DT block required"

            logger.info(message)
        if flags & 1 * 8:
            message = f"Unfinalised file {self.name}: Update of length for last RD block required"

            logger.info(message)
        if flags & 1 << 4:
            message = (
                f"Unfinalised file {self.name}:"
                " Update of last DL block in each chained list"
                " of DL blocks required"
            )

            logger.info(message)
        if flags & 1 << 5:
            message = (
                f"Unfinalised file {self.name}:"
                " Update of cg_data_bytes and cg_inval_bytes"
                " in VLSD CG block required"
            )

            logger.info(message)
        if flags & 1 << 6:
            message = (
                f"Unfinalised file {self.name}:"
                " Update of offset values for VLSD channel required"
                " in case a VLSD CG block is used"
            )

            logger.info(message)

        return flags

    def _read(self, mapped=False):

        stream = self._file
        self._mapped = mapped
        dg_cntr = 0

        stream.seek(0, 2)
        self.file_limit = stream.tell()
        stream.seek(0)

        cg_count, _ = count_channel_groups(stream)
        if self._callback:
            self._callback(0, cg_count)
        current_cg_index = 0

        self.identification = FileIdentificationBlock(stream=stream, mapped=mapped)
        version = self.identification["version_str"]
        self.version = version.decode("utf-8").strip(" \n\t\0")

        if self.version >= "4.10":
            # Check for finalization past version 4.10
            finalisation_flags = self._check_finalised()

            if finalisation_flags:
                message = f"Attempting finalization of {self.name}"
                logger.info(message)
                self._finalize()
                self._mapped = mapped = False

        stream = self._file

        self.header = HeaderBlock(address=0x40, stream=stream, mapped=mapped)

        # read file history
        fh_addr = self.header["file_history_addr"]
        while fh_addr:
            if fh_addr > self.file_limit:
                logger.warning(
                    f"File history address {fh_addr:X} is outside the file size {self.file_limit}"
                )
                break
            history_block = FileHistory(address=fh_addr, stream=stream, mapped=mapped)
            self.file_history.append(history_block)
            fh_addr = history_block.next_fh_addr

        # read attachments
        at_addr = self.header["first_attachment_addr"]
        index = 0
        while at_addr:
            if at_addr > self.file_limit:
                logger.warning(
                    f"Attachment address {at_addr:X} is outside the file size {self.file_limit}"
                )
                break
            at_block = AttachmentBlock(address=at_addr, stream=stream, mapped=mapped)
            self._attachments_map[at_addr] = index
            self.attachments.append(at_block)
            at_addr = at_block.next_at_addr
            index += 1

        # go to first date group and read each data group sequentially
        dg_addr = self.header.first_dg_addr

        while dg_addr:
            if dg_addr > self.file_limit:
                logger.warning(
                    f"Data group address {dg_addr:X} is outside the file size {self.file_limit}"
                )
                break
            new_groups = []
            group = DataGroup(address=dg_addr, stream=stream, mapped=mapped)
            record_id_nr = group.record_id_len

            # go to first channel group of the current data group
            cg_addr = first_cg_addr = group.first_cg_addr

            cg_nr = 0

            cg_size = {}

            while cg_addr:
                if cg_addr > self.file_limit:
                    logger.warning(
                        f"Channel group address {cg_addr:X} is outside the file size {self.file_limit}"
                    )
                    break
                cg_nr += 1

                if cg_addr == first_cg_addr:
                    grp = Group(group)
                else:
                    grp = Group(group.copy())

                # read each channel group sequentially
                block = ChannelGroup(
                    address=cg_addr,
                    stream=stream,
                    mapped=mapped,
                    si_map=self._si_map,
                    version=self.version,
                    tx_map=self._interned_strings,
                )
                self._cg_map[cg_addr] = dg_cntr
                channel_group = grp.channel_group = block

                grp.record_size = cg_size

                if channel_group.flags & v4c.FLAG_CG_VLSD:
                    # VLDS flag
                    record_id = channel_group.record_id
                    cg_size[record_id] = 0
                elif channel_group.flags & v4c.FLAG_CG_BUS_EVENT:
                    samples_size = channel_group.samples_byte_nr
                    inval_size = channel_group.invalidation_bytes_nr
                    record_id = channel_group.record_id
                    cg_size[record_id] = samples_size + inval_size
                else:
                    # in case no `cg_flags` are set
                    samples_size = channel_group.samples_byte_nr
                    inval_size = channel_group.invalidation_bytes_nr
                    record_id = channel_group.record_id
                    cg_size[record_id] = samples_size + inval_size

                if record_id_nr:
                    grp.sorted = False
                else:
                    grp.sorted = True

                # go to first channel of the current channel group
                ch_addr = channel_group.first_ch_addr
                ch_cntr = 0

                # Read channels by walking recursively in the channel group
                # starting from the first channel
                self._read_channels(
                    ch_addr, grp, stream, dg_cntr, ch_cntr, mapped=mapped
                )

                cg_addr = channel_group.next_cg_addr

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

            address = group.data_block_addr

            total_size = 0
            inval_total_size = 0
            block_type = b"##DT"

            for new_group in new_groups:

                channel_group = new_group.channel_group
                if channel_group.flags & v4c.FLAG_CG_REMOTE_MASTER:
                    block_type = b"##DV"
                    total_size += (
                        channel_group.samples_byte_nr * channel_group.cycles_nr
                    )
                    inval_total_size += (
                        channel_group.invalidation_bytes_nr * channel_group.cycles_nr
                    )
                else:
                    block_type = b"##DT"
                    total_size += (
                        channel_group.samples_byte_nr
                        + channel_group.invalidation_bytes_nr
                    ) * channel_group.cycles_nr

            if (
                self.identification["unfinalized_standard_flags"]
                & v4c.FLAG_UNFIN_UPDATE_CG_COUNTER
            ):
                total_size = int(10 ** 12)
                inval_total_size = int(10 ** 12)

            info, uses_ld = self._get_data_blocks_info(
                address=address,
                stream=stream,
                block_type=block_type,
                mapped=mapped,
                total_size=total_size,
                inval_total_size=inval_total_size,
            )

            for grp in new_groups:
                grp.data_location = v4c.LOCATION_ORIGINAL_FILE
                grp.set_blocks_info(info)
                grp.uses_ld = uses_ld

            self.groups.extend(new_groups)

            dg_addr = group.next_dg_addr

        # all channels have been loaded so now we can link the
        # channel dependencies and load the signal data for VLSD channels
        for gp_index, grp in enumerate(self.groups):

            if (
                self.version >= "4.20"
                and grp.channel_group.flags & v4c.FLAG_CG_REMOTE_MASTER
            ):
                grp.channel_group.cg_master_index = self._cg_map[
                    grp.channel_group.cg_master_addr
                ]
                index = grp.channel_group.cg_master_index

            else:
                index = gp_index

            self.virtual_groups_map[gp_index] = index
            if index not in self.virtual_groups:
                self.virtual_groups[index] = VirtualChannelGroup()

            virtual_channel_group = self.virtual_groups[index]
            virtual_channel_group.groups.append(gp_index)
            virtual_channel_group.record_size += (
                grp.channel_group.samples_byte_nr
                + grp.channel_group.invalidation_bytes_nr
            )
            virtual_channel_group.cycles_nr = grp.channel_group.cycles_nr

            for ch_index, dep_list in enumerate(grp.channel_dependencies):
                if not dep_list:
                    continue

                for dep in dep_list:
                    if isinstance(dep, ChannelArrayBlock):

                        if dep.flags & v4c.FLAG_CA_DYNAMIC_AXIS:
                            for i in range(dep.dims):
                                ch_addr = dep[f"dynamic_size_{i}_ch_addr"]
                                if ch_addr:
                                    ref_channel = self._ch_map[ch_addr]
                                    dep.dynamic_size_channels.append(ref_channel)
                                else:
                                    dep.dynamic_size_channels.append(None)

                        if dep.flags & v4c.FLAG_CA_INPUT_QUANTITY:
                            for i in range(dep.dims):
                                ch_addr = dep[f"input_quantity_{i}_ch_addr"]
                                if ch_addr:
                                    ref_channel = self._ch_map[ch_addr]
                                    dep.input_quantity_channels.append(ref_channel)
                                else:
                                    dep.input_quantity_channels.append(None)

                        if dep.flags & v4c.FLAG_CA_OUTPUT_QUANTITY:
                            ch_addr = dep["output_quantity_ch_addr"]
                            if ch_addr:
                                ref_channel = self._ch_map[ch_addr]
                                dep.output_quantity_channel = ref_channel
                            else:
                                dep.output_quantity_channel = None

                        if dep.flags & v4c.FLAG_CA_COMPARISON_QUANTITY:
                            ch_addr = dep["comparison_quantity_ch_addr"]
                            if ch_addr:
                                ref_channel = self._ch_map[ch_addr]
                                dep.comparison_quantity_channel = ref_channel
                            else:
                                dep.comparison_quantity_channel = None

                        if dep.flags & v4c.FLAG_CA_AXIS:
                            for i in range(dep.dims):
                                cc_addr = dep[f"axis_conversion_{i}"]
                                if cc_addr:
                                    conv = ChannelConversion(
                                        stream=stream,
                                        address=cc_addr,
                                        mapped=mapped,
                                        tx_map={},
                                    )
                                    dep.axis_conversions.append(conv)
                                else:
                                    dep.axis_conversions.append(None)

                        if (dep.flags & v4c.FLAG_CA_AXIS) and not (
                            dep.flags & v4c.FLAG_CA_FIXED_AXIS
                        ):
                            for i in range(dep.dims):
                                ch_addr = dep[f"scale_axis_{i}_ch_addr"]
                                if ch_addr:
                                    ref_channel = self._ch_map[ch_addr]
                                    dep.axis_channels.append(ref_channel)
                                else:
                                    dep.axis_channels.append(None)
                    else:
                        break

        self._sort()

        for i, gp in enumerate(self.groups):
            for j, ch in enumerate(gp.channels):
                if isinstance(gp.signal_data[j], int):
                    gp.signal_data[j] = self._get_signal_data_blocks_info(
                        gp.signal_data[j], stream
                    )

        for grp in self.groups:
            channels = grp.channels
            if (
                len(channels) == 1
                and channels[0].dtype_fmt.itemsize == grp.channel_group.samples_byte_nr
            ):
                grp.single_channel_dtype = channels[0].dtype_fmt

        self._process_bus_logging()

        # read events
        addr = self.header.first_event_addr
        ev_map = {}
        event_index = 0
        while addr:
            if addr > self.file_limit:
                logger.warning(
                    f"Event address {addr:X} is outside the file size {self.file_limit}"
                )
                break
            event = EventBlock(address=addr, stream=stream, mapped=mapped)
            event.update_references(self._ch_map, self._cg_map)
            self.events.append(event)
            ev_map[addr] = event_index
            event_index += 1

            addr = event.next_ev_addr

        for event in self.events:
            addr = event.parent_ev_addr
            if addr:
                parent = ev_map.get(addr, None)
                if parent is not None:
                    event.parent = parent
                else:
                    event.parent = None

            addr = event.range_start_ev_addr
            if addr:
                range_start_ev_addr = ev_map.get(addr, None)
                if range_start_ev_addr is not None:
                    event.parent = range_start_ev_addr
                else:
                    event.parent = None

        self._si_map.clear()
        self._ch_map.clear()
        self._cc_map.clear()

        self._interned_strings.clear()
        self._attachments_map.clear()

        self.progress = cg_count, cg_count

    def _read_channels(
        self,
        ch_addr,
        grp,
        stream,
        dg_cntr,
        ch_cntr,
        channel_composition=False,
        mapped=False,
    ):

        filter_channels = self.use_load_filter
        use_display_names = self._use_display_names

        channels = grp.channels

        dependencies = grp.channel_dependencies

        unique_names = UniqueDB()

        if channel_composition:
            composition = []
            composition_channels = []

        if grp.channel_group.path_separator:
            path_separator = chr(grp.channel_group.path_separator)
        else:
            path_separator = "\\"

        while ch_addr:
            # read channel block and create channel object

            if ch_addr > self.file_limit:
                logger.warning(
                    f"Channel address {ch_addr:X} is outside the file size {self.file_limit}"
                )
                break

            if filter_channels:
                if mapped:
                    (
                        id_,
                        links_nr,
                        next_ch_addr,
                        name_addr,
                        comment_addr,
                    ) = v4c.CHANNEL_FILTER_uf(stream, ch_addr)
                    channel_type = stream[ch_addr + v4c.COMMON_SIZE + links_nr * 8]
                    name = get_text_v4(name_addr, stream, mapped=mapped)
                    if use_display_names:
                        comment = get_text_v4(comment_addr, stream, mapped=mapped)
                        display_name = extract_display_name(comment)
                    else:
                        display_name = ""
                        comment = None

                else:
                    stream.seek(ch_addr)
                    (
                        id_,
                        links_nr,
                        next_ch_addr,
                        name_addr,
                        comment_addr,
                    ) = v4c.CHANNEL_FILTER_u(stream.read(v4c.CHANNEL_FILTER_SIZE))
                    stream.seek(ch_addr + v4c.COMMON_SIZE + links_nr * 8)
                    channel_type = stream.read(1)[0]
                    name = get_text_v4(name_addr, stream, mapped=mapped)

                    if use_display_names:
                        comment = get_text_v4(comment_addr, stream, mapped=mapped)
                        display_name = extract_display_name(comment)
                    else:
                        display_name = ""
                        comment = None

                if id_ != b"##CN":
                    message = f'Expected "##CN" block @{hex(ch_addr)} but found "{id_}"'
                    raise MdfException(message)

                if (
                    channel_composition
                    or channel_type in v4c.MASTER_TYPES
                    or name in self.load_filter
                    or (use_display_names and display_name in self.load_filter)
                ):
                    if comment is None:
                        comment = get_text_v4(comment_addr, stream, mapped=mapped)
                    channel = Channel(
                        address=ch_addr,
                        stream=stream,
                        cc_map=self._cc_map,
                        si_map=self._si_map,
                        at_map=self._attachments_map,
                        use_display_names=use_display_names,
                        mapped=mapped,
                        tx_map=self._interned_strings,
                        file_limit=self.file_limit,
                        parsed_strings=(name, display_name, comment),
                    )

                else:
                    ch_addr = next_ch_addr
                    continue
            else:
                channel = Channel(
                    address=ch_addr,
                    stream=stream,
                    cc_map=self._cc_map,
                    si_map=self._si_map,
                    at_map=self._attachments_map,
                    use_display_names=use_display_names,
                    mapped=mapped,
                    tx_map=self._interned_strings,
                    file_limit=self.file_limit,
                    parsed_strings=None,
                )

            if channel.channel_type == v4c.CHANNEL_TYPE_SYNC:
                channel.attachment = self._attachments_map.get(
                    channel.data_block_addr,
                    None,
                )

            if self._remove_source_from_channel_names:
                channel.name = channel.name.split(path_separator, 1)[0]
                channel.display_name = channel.display_name.split(path_separator, 1)[0]

            entry = (dg_cntr, ch_cntr)
            self._ch_map[ch_addr] = entry

            channels.append(channel)
            if channel_composition:
                composition.append(entry)
                composition_channels.append(channel)

            if channel.display_name:
                self.channels_db.add(channel.display_name, entry)
            self.channels_db.add(channel.name, entry)

            # signal data
            cn_data_addr = channel.data_block_addr
            grp.signal_data.append(cn_data_addr)
            if cn_data_addr:
                self._cn_data_map[cn_data_addr] = entry

            if channel.channel_type in MASTER_CHANNELS:
                self.masters_db[dg_cntr] = ch_cntr

            ch_cntr += 1

            component_addr = channel.component_addr

            if component_addr:

                if component_addr > self.file_limit:
                    logger.warning(
                        f"Channel component address {component_addr:X} is outside the file size {self.file_limit}"
                    )
                    break
                # check if it is a CABLOCK or CNBLOCK
                stream.seek(component_addr)
                blk_id = stream.read(4)
                if blk_id == b"##CN":
                    index = ch_cntr - 1
                    dependencies.append(None)
                    (
                        ch_cntr,
                        ret_composition,
                        ret_composition_dtype,
                    ) = self._read_channels(
                        component_addr,
                        grp,
                        stream,
                        dg_cntr,
                        ch_cntr,
                        True,
                        mapped=mapped,
                    )
                    dependencies[index] = ret_composition

                    channel.dtype_fmt = ret_composition_dtype

                else:
                    # only channel arrays with storage=CN_TEMPLATE are
                    # supported so far
                    first_dep = ca_block = ChannelArrayBlock(
                        address=component_addr, stream=stream, mapped=mapped
                    )
                    if ca_block.storage != v4c.CA_STORAGE_TYPE_CN_TEMPLATE:
                        logger.warning("Only CN template arrays are supported")
                    ca_list = [ca_block]
                    while ca_block.composition_addr:
                        ca_block = ChannelArrayBlock(
                            address=ca_block.composition_addr,
                            stream=stream,
                            mapped=mapped,
                        )
                        ca_list.append(ca_block)
                    dependencies.append(ca_list)

                    channel.dtype_fmt = dtype(
                        get_fmt_v4(
                            channel.data_type,
                            channel.bit_offset + channel.bit_count,
                            channel.channel_type,
                        )
                    )

            else:
                dependencies.append(None)

                channel.dtype_fmt = dtype(
                    get_fmt_v4(
                        channel.data_type,
                        channel.bit_offset + channel.bit_count,
                        channel.channel_type,
                    )
                )

            # go to next channel of the current channel group
            ch_addr = channel.next_ch_addr

        if channel_composition:
            composition_channels.sort()
            composition_dtype = dtype(
                [
                    (unique_names.get_unique_name(channel.name), channel.dtype_fmt)
                    for channel in composition_channels
                ]
            )

        else:
            composition = None
            composition_dtype = None

        return ch_cntr, composition, composition_dtype

    def _load_signal_data(
        self, group=None, index=None, start_offset=None, end_offset=None
    ):
        """this method is used to get the channel signal data, usually for
        VLSD channels

        Parameters
        ----------
        address : int
            address of referenced block
        stream : handle
            file IO stream handle

        Returns
        -------
        data : bytes
            signal data bytes

        """

        data = []

        if group is not None and index is not None:
            info_blocks = group.signal_data[index]

            if info_blocks is not None:

                if start_offset is None and end_offset is None:
                    for info in info_blocks:
                        address, original_size, compressed_size, block_type, param = (
                            info.address,
                            info.original_size,
                            info.compressed_size,
                            info.block_type,
                            info.param,
                        )

                        if not info.original_size:
                            continue
                        if info.location == v4c.LOCATION_TEMPORARY_FILE:
                            stream = self._tempfile
                        else:
                            stream = self._file

                        stream.seek(address)
                        new_data = stream.read(compressed_size)
                        if block_type == v4c.DZ_BLOCK_DEFLATE:
                            new_data = decompress(new_data, 0, original_size)
                        elif block_type == v4c.DZ_BLOCK_TRANSPOSED:
                            new_data = decompress(new_data, 0, original_size)
                            cols = param
                            lines = original_size // cols

                            nd = frombuffer(new_data[: lines * cols], dtype=uint8)
                            nd = nd.reshape((cols, lines))
                            new_data = nd.T.tobytes() + new_data[lines * cols :]
                        elif block_type == v4c.DZ_BLOCK_LZ:
                            new_data = lz_decompress(new_data)

                        data.append(new_data)

                else:
                    start_offset = int(start_offset)
                    end_offset = int(end_offset)

                    current_offset = 0

                    for info in info_blocks:
                        address, original_size, compressed_size, block_type, param = (
                            info.address,
                            info.original_size,
                            info.compressed_size,
                            info.block_type,
                            info.param,
                        )

                        if not info.original_size:
                            continue
                        if info.location == v4c.LOCATION_TEMPORARY_FILE:
                            stream = self._tempfile
                        else:
                            stream = self._file

                        if current_offset + original_size < start_offset:
                            current_offset += original_size
                            continue

                        stream.seek(address)
                        new_data = stream.read(compressed_size)
                        if block_type == v4c.DZ_BLOCK_DEFLATE:
                            new_data = decompress(new_data, 0, original_size)
                        elif block_type == v4c.DZ_BLOCK_TRANSPOSED:
                            new_data = decompress(new_data, 0, original_size)
                            cols = param
                            lines = original_size // cols

                            nd = frombuffer(new_data[: lines * cols], dtype=uint8)
                            nd = nd.reshape((cols, lines))
                            new_data = nd.T.tobytes() + new_data[lines * cols :]
                        elif block_type == v4c.DZ_BLOCK_LZ:
                            new_data = lz_decompress(new_data)

                        if current_offset + original_size > end_offset:
                            start_index = max(0, start_offset - current_offset)
                            (last_sample_size,) = UINT32_uf(
                                new_data, end_offset - current_offset
                            )
                            data.append(
                                new_data[
                                    start_index : end_offset
                                    - current_offset
                                    + last_sample_size
                                    + 4
                                ]
                            )
                            break

                        else:
                            if start_offset > current_offset:
                                data.append(new_data[start_offset - current_offset :])
                            else:
                                data.append(new_data)

                            current_offset += original_size

                data = b"".join(data)
            else:
                data = b""
        else:
            data = b""

        return data

    def _load_data(
        self, group, record_offset=0, record_count=None, optimize_read=False
    ):
        """ get group's data block bytes """

        offset = 0
        invalidation_offset = 0
        has_yielded = False
        _count = 0
        data_group = group.data_group
        channel_group = group.channel_group

        if group.data_location == v4c.LOCATION_ORIGINAL_FILE:
            stream = self._file
        else:
            stream = self._tempfile

        read = stream.read
        seek = stream.seek

        if group.uses_ld:
            samples_size = channel_group.samples_byte_nr
            invalidation_size = channel_group.invalidation_bytes_nr
            invalidation_record_offset = record_offset * invalidation_size
            rm = True
        else:
            rm = False
            samples_size = (
                channel_group.samples_byte_nr + channel_group.invalidation_bytes_nr
            )
            invalidation_size = channel_group.invalidation_bytes_nr

        record_offset *= samples_size

        finished = False
        if record_count is not None:
            invalidation_record_count = record_count * invalidation_size
            record_count *= samples_size

        if not samples_size:
            if rm:
                yield b"", offset, _count, b""
            else:
                yield b"", offset, _count, None
        else:

            if group.read_split_count:
                split_size = group.read_split_count * samples_size
                invalidation_split_size = group.read_split_count * invalidation_size
            else:

                if self._read_fragment_size:
                    split_size = self._read_fragment_size // samples_size
                    invalidation_split_size = split_size * invalidation_size
                    split_size *= samples_size

                else:
                    channels_nr = len(group.channels)

                    y_axis = CONVERT

                    idx = searchsorted(CHANNEL_COUNT, channels_nr, side="right") - 1
                    if idx < 0:
                        idx = 0
                    split_size = y_axis[idx]

                    split_size = split_size // samples_size
                    invalidation_split_size = split_size * invalidation_size
                    split_size *= samples_size

            if split_size == 0:
                split_size = samples_size
                invalidation_split_size = invalidation_size

            split_size = int(split_size)
            invalidation_split_size = int(invalidation_split_size)

            blocks = iter(group.data_blocks)

            if group.data_blocks:

                cur_size = 0
                data = []

                cur_invalidation_size = 0
                invalidation_data = []

                while True:
                    try:
                        info = next(blocks)
                        (
                            address,
                            original_size,
                            compressed_size,
                            block_type,
                            param,
                            block_limit,
                        ) = (
                            info.address,
                            info.original_size,
                            info.compressed_size,
                            info.block_type,
                            info.param,
                            info.block_limit,
                        )

                        if rm and invalidation_size:
                            invalidation_info = info.invalidation_block
                        else:
                            invalidation_info = None
                    except StopIteration:
                        break

                    if offset + original_size < record_offset + 1:
                        offset += original_size
                        if rm and invalidation_size:
                            if invalidation_info.all_valid:
                                count = original_size // samples_size
                                invalidation_offset += count * invalidation_size
                            else:
                                invalidation_offset += invalidation_info.original_size
                        continue

                    seek(address)
                    new_data = read(compressed_size)

                    if block_type == v4c.DZ_BLOCK_DEFLATE:
                        new_data = decompress(new_data, 0, original_size)
                    elif block_type == v4c.DZ_BLOCK_TRANSPOSED:
                        new_data = decompress(new_data, 0, original_size)
                        cols = param
                        lines = original_size // cols

                        nd = frombuffer(new_data[: lines * cols], dtype=uint8)
                        nd = nd.reshape((cols, lines))
                        new_data = nd.T.tobytes() + new_data[lines * cols :]
                    elif block_type == v4c.DZ_BLOCK_LZ:
                        new_data = lz_decompress(new_data)

                    if block_limit is not None:
                        new_data = new_data[:block_limit]

                    if len(data) > split_size - cur_size:
                        new_data = memoryview(new_data)

                    if rm and invalidation_size:

                        if invalidation_info.all_valid:
                            count = original_size // samples_size
                            new_invalidation_data = bytes(count * invalidation_size)

                        else:
                            seek(invalidation_info.address)
                            new_invalidation_data = read(invalidation_info.size)
                            if invalidation_info.block_type == v4c.DZ_BLOCK_DEFLATE:
                                new_invalidation_data = decompress(
                                    new_invalidation_data,
                                    0,
                                    invalidation_info.original_size,
                                )
                            elif (
                                invalidation_info.block_type == v4c.DZ_BLOCK_TRANSPOSED
                            ):
                                new_invalidation_data = decompress(
                                    new_invalidation_data,
                                    0,
                                    invalidation_info.original_size,
                                )
                                cols = invalidation_info.param
                                lines = invalidation_info.original_size // cols

                                nd = frombuffer(
                                    new_invalidation_data[: lines * cols], dtype=uint8
                                )
                                nd = nd.reshape((cols, lines))
                                new_invalidation_data = (
                                    nd.T.tobytes()
                                    + new_invalidation_data[lines * cols :]
                                )
                            if invalidation_info.block_limit is not None:
                                new_invalidation_data = new_invalidation_data[
                                    : invalidation_info.block_limit
                                ]

                        inv_size = len(new_invalidation_data)

                    if offset < record_offset:
                        delta = record_offset - offset
                        new_data = new_data[delta:]
                        original_size -= delta
                        offset = record_offset

                        if rm and invalidation_size:
                            delta = invalidation_record_offset - invalidation_offset
                            new_invalidation_data = new_invalidation_data[delta:]
                            inv_size -= delta
                            invalidation_offset = invalidation_record_offset

                    while original_size >= split_size - cur_size:
                        if data:
                            data.append(new_data[: split_size - cur_size])
                            new_data = new_data[split_size - cur_size :]
                            data_ = b"".join(data)

                            if rm and invalidation_size:
                                invalidation_data.append(
                                    new_invalidation_data[
                                        : invalidation_split_size
                                        - cur_invalidation_size
                                    ]
                                )
                                new_invalidation_data = new_invalidation_data[
                                    invalidation_split_size - cur_invalidation_size :
                                ]
                                invalidation_data_ = b"".join(invalidation_data)

                            if record_count is not None:
                                if rm and invalidation_size:
                                    __data = data_[:record_count]
                                    _count = len(__data) // samples_size
                                    yield __data, offset // samples_size, _count, invalidation_data_[
                                        :invalidation_record_count
                                    ]
                                    invalidation_record_count -= len(invalidation_data_)
                                else:
                                    __data = data_[:record_count]
                                    _count = len(__data) // samples_size
                                    yield __data, offset // samples_size, _count, None
                                has_yielded = True
                                record_count -= len(data_)
                                if record_count <= 0:
                                    finished = True
                                    break
                            else:
                                if rm and invalidation_size:
                                    _count = len(data_) // samples_size
                                    yield data_, offset // samples_size, _count, invalidation_data_
                                else:
                                    _count = len(data_) // samples_size
                                    yield data_, offset // samples_size, _count, None
                                has_yielded = True

                            data = []

                        else:
                            data_, new_data = (
                                new_data[:split_size],
                                new_data[split_size:],
                            )
                            if rm and invalidation_size:
                                invalidation_data_ = new_invalidation_data[
                                    :invalidation_split_size
                                ]
                                new_invalidation_data = new_invalidation_data[
                                    invalidation_split_size:
                                ]

                            if record_count is not None:
                                if rm and invalidation_size:
                                    yield data_[
                                        :record_count
                                    ], offset // samples_size, _count, invalidation_data_[
                                        :invalidation_record_count
                                    ]
                                    invalidation_record_count -= len(invalidation_data_)
                                else:
                                    __data = data_[:record_count]
                                    _count = len(__data) // samples_size
                                    yield __data, offset // samples_size, _count, None
                                has_yielded = True
                                record_count -= len(data_)
                                if record_count <= 0:
                                    finished = True
                                    break
                            else:
                                if rm and invalidation_size:
                                    _count = len(data_) // samples_size
                                    yield data_, offset // samples_size, _count, invalidation_data_
                                else:
                                    _count = len(data_) // samples_size
                                    yield data_, offset // samples_size, _count, None
                                has_yielded = True

                        offset += split_size
                        original_size -= split_size - cur_size
                        data = []
                        cur_size = 0

                        if rm and invalidation_size:
                            invalidation_offset += invalidation_split_size
                            invalidation_data = []
                            cur_invalidation_size = 0
                            inv_size -= invalidation_split_size - cur_invalidation_size

                    if finished:
                        data = []
                        if rm and invalidation_size:
                            invalidation_data = []
                        break

                    if original_size:
                        data.append(new_data)
                        cur_size += original_size
                        original_size = 0

                        if rm and invalidation_size:
                            invalidation_data.append(new_invalidation_data)
                            cur_invalidation_size += inv_size

                if data:
                    data_ = b"".join(data)
                    if rm and invalidation_size:
                        invalidation_data_ = b"".join(invalidation_data)
                    if record_count is not None:
                        if rm and invalidation_size:
                            __data = data_[:record_count]
                            _count = len(__data) // samples_size
                            yield __data, offset // samples_size, _count, invalidation_data_[
                                :invalidation_record_count
                            ]
                            invalidation_record_count -= len(invalidation_data_)
                        else:
                            __data = data_[:record_count]
                            _count = len(__data) // samples_size
                            yield __data, offset // samples_size, _count, None
                        has_yielded = True
                        record_count -= len(data_)
                    else:
                        if rm and invalidation_size:
                            _count = len(data_) // samples_size
                            yield data_, offset // samples_size, _count, invalidation_data_
                        else:
                            _count = len(data_) // samples_size
                            yield data_, offset // samples_size, _count, None
                        has_yielded = True
                    data = []

                if not has_yielded:
                    if rm and invalidation_size:
                        yield b"", 0, 0, b""
                    else:
                        yield b"", 0, 0, None
            else:
                if rm and invalidation_size:
                    yield b"", offset, 0, b""
                else:
                    yield b"", offset, 0, None

    def _prepare_record(self, group):
        """compute record dtype and parents dict for this group

        Parameters
        ----------
        group : dict
            MDF group dict

        Returns
        -------
        parents, dtypes : dict, numpy.dtype
            mapping of channels to records fields, records fields dtype

        """

        parents, dtypes = group.parents, group.types

        if parents is None:
            no_parent = None, None
            channel_group = group.channel_group
            channels = group.channels

            record_size = channel_group.samples_byte_nr
            invalidation_bytes_nr = channel_group.invalidation_bytes_nr
            next_byte_aligned_position = 0
            types = []
            current_parent = ""
            parent_start_offset = 0
            parents = {}
            group_channels = UniqueDB()

            sortedchannels = sorted(enumerate(channels), key=lambda i: i[1])
            for original_index, new_ch in sortedchannels:
                start_offset = new_ch.byte_offset
                bit_offset = new_ch.bit_offset
                data_type = new_ch.data_type
                bit_count = new_ch.bit_count
                ch_type = new_ch.channel_type
                dependency_list = group.channel_dependencies[original_index]
                name = new_ch.name

                # handle multiple occurrence of same channel name
                name = group_channels.get_unique_name(name)

                if start_offset >= next_byte_aligned_position:
                    if ch_type not in v4c.VIRTUAL_TYPES:
                        if not dependency_list:
                            parent_start_offset = start_offset

                            # check if there are byte gaps in the record
                            gap = parent_start_offset - next_byte_aligned_position
                            if gap:
                                types.append(("", f"V{gap}"))

                            # adjust size to 1, 2, 4 or 8 bytes
                            size = bit_offset + bit_count

                            byte_size, rem = size // 8, size % 8
                            if rem:
                                byte_size += 1
                            bit_size = byte_size * 8

                            if data_type in (
                                v4c.DATA_TYPE_SIGNED_MOTOROLA,
                                v4c.DATA_TYPE_UNSIGNED_MOTOROLA,
                            ):
                                if size > 32:
                                    size = 8
                                    bit_offset += 64 - bit_size
                                elif size > 16:
                                    size = 4
                                    bit_offset += 32 - bit_size
                                elif size > 8:
                                    size = 2
                                    bit_offset += 16 - bit_size
                                else:
                                    size = 1
                            elif data_type not in v4c.NON_SCALAR_TYPES:
                                if size > 32:
                                    size = 8
                                elif size > 16:
                                    size = 4
                                elif size > 8:
                                    size = 2
                                else:
                                    size = 1
                            else:
                                size = size // 8

                            next_byte_aligned_position = parent_start_offset + size
                            bit_count = size * 8
                            if next_byte_aligned_position <= record_size:
                                if not new_ch.dtype_fmt:
                                    new_ch.dtype_fmt = get_fmt_v4(
                                        data_type, bit_count, ch_type
                                    )
                                dtype_pair = (name, new_ch.dtype_fmt)
                                types.append(dtype_pair)
                                parents[original_index] = name, bit_offset
                            else:
                                next_byte_aligned_position = parent_start_offset

                            current_parent = name
                        else:
                            parents[original_index] = no_parent

                    # virtual channels do not have bytes in the record
                    else:
                        parents[original_index] = no_parent

                else:
                    if not dependency_list:
                        size = bit_offset + bit_count
                        byte_size, rem = size // 8, size % 8
                        if rem:
                            byte_size += 1

                        max_overlapping_size = (
                            next_byte_aligned_position - start_offset
                        ) * 8
                        needed_size = bit_offset + bit_count

                        if max_overlapping_size >= needed_size:
                            if data_type in (
                                v4c.DATA_TYPE_SIGNED_MOTOROLA,
                                v4c.DATA_TYPE_UNSIGNED_MOTOROLA,
                            ):
                                parents[original_index] = (
                                    current_parent,
                                    (
                                        next_byte_aligned_position
                                        - start_offset
                                        - byte_size
                                    )
                                    * 8
                                    + bit_offset,
                                )
                            else:
                                parents[original_index] = (
                                    current_parent,
                                    ((start_offset - parent_start_offset) * 8)
                                    + bit_offset,
                                )
                    else:
                        parents[original_index] = no_parent
                if next_byte_aligned_position > record_size:
                    break

            gap = record_size - next_byte_aligned_position
            if gap > 0:
                dtype_pair = "", f"V{gap}"
                types.append(dtype_pair)

            if not group.uses_ld:

                dtype_pair = "invalidation_bytes", "<u1", (invalidation_bytes_nr,)
                types.append(dtype_pair)

            dtypes = dtype(types)

            group.parents, group.types = parents, dtypes

        return parents, dtypes

    def _get_data_blocks_info(
        self,
        address,
        stream,
        block_type=b"##DT",
        mapped=False,
        total_size=0,
        inval_total_size=0,
    ):
        info = []
        mapped = mapped or not is_file_like(stream)
        uses_ld = False

        if mapped:
            if address:
                id_string, block_len = COMMON_SHORT_uf(stream, address)

                # can be a DataBlock
                if id_string == block_type:
                    size = block_len - 24
                    if size:
                        if total_size < size:
                            block_limit = total_size
                        else:
                            block_limit = None
                        total_size -= size
                        info.append(
                            DataBlockInfo(
                                address=address + COMMON_SIZE,
                                block_type=v4c.DT_BLOCK,
                                original_size=size,
                                compressed_size=size,
                                param=0,
                                block_limit=block_limit,
                            )
                        )
                # or a DataZippedBlock
                elif id_string == b"##DZ":
                    (
                        original_type,
                        zip_type,
                        param,
                        original_size,
                        zip_size,
                    ) = v4c.DZ_COMMON_INFO_uf(stream, address)

                    if original_size:
                        if zip_type == v4c.FLAG_DZ_DEFLATE:
                            block_type_ = v4c.DZ_BLOCK_DEFLATE
                            param = 0
                        else:
                            block_type_ = v4c.DZ_BLOCK_TRANSPOSED
                        if total_size < original_size:
                            block_limit = total_size
                        else:
                            block_limit = None
                        total_size -= original_size
                        info.append(
                            DataBlockInfo(
                                address=address + v4c.DZ_COMMON_SIZE,
                                block_type=block_type_,
                                original_size=original_size,
                                compressed_size=zip_size,
                                param=param,
                                block_limit=block_limit,
                            )
                        )

                # or a DataList
                elif id_string == b"##DL":
                    while address:
                        dl = DataList(address=address, stream=stream, mapped=mapped)
                        for i in range(dl.data_block_nr):
                            addr = dl[f"data_block_addr{i}"]

                            id_string, block_len = COMMON_SHORT_uf(stream, addr)
                            # can be a DataBlock
                            if id_string == block_type:
                                size = block_len - 24
                                if size:
                                    if total_size < size:
                                        block_limit = total_size
                                    else:
                                        block_limit = None
                                    total_size -= size
                                    info.append(
                                        DataBlockInfo(
                                            address=addr + COMMON_SIZE,
                                            block_type=v4c.DT_BLOCK,
                                            original_size=size,
                                            compressed_size=size,
                                            param=0,
                                            block_limit=block_limit,
                                        )
                                    )
                            # or a DataZippedBlock
                            elif id_string == b"##DZ":
                                (
                                    original_type,
                                    zip_type,
                                    param,
                                    original_size,
                                    zip_size,
                                ) = v4c.DZ_COMMON_INFO_uf(stream, addr)

                                if original_size:
                                    if zip_type == v4c.FLAG_DZ_DEFLATE:
                                        block_type_ = v4c.DZ_BLOCK_DEFLATE
                                        param = 0
                                    else:
                                        block_type_ = v4c.DZ_BLOCK_TRANSPOSED
                                    if total_size < original_size:
                                        block_limit = total_size
                                    else:
                                        block_limit = None
                                    total_size -= original_size
                                    info.append(
                                        DataBlockInfo(
                                            address=addr + v4c.DZ_COMMON_SIZE,
                                            block_type=block_type_,
                                            original_size=original_size,
                                            compressed_size=zip_size,
                                            param=param,
                                            block_limit=block_limit,
                                        )
                                    )
                        address = dl.next_dl_addr

                # or a ListData
                elif id_string == b"##LD":
                    uses_ld = True
                    while address:
                        ld = ListData(address=address, stream=stream, mapped=mapped)
                        has_invalidation = ld.flags & v4c.FLAG_LD_INVALIDATION_PRESENT
                        for i in range(ld.data_block_nr):
                            addr = ld[f"data_block_addr_{i}"]

                            id_string, block_len = COMMON_SHORT_uf(stream, addr)
                            # can be a DataBlock
                            if id_string == b"##DV":
                                size = block_len - 24
                                if size:
                                    if total_size < size:
                                        block_limit = total_size
                                    else:
                                        block_limit = None
                                    total_size -= size
                                    info.append(
                                        DataBlockInfo(
                                            address=addr + COMMON_SIZE,
                                            block_type=v4c.DT_BLOCK,
                                            original_size=size,
                                            compressed_size=size,
                                            param=0,
                                            block_limit=block_limit,
                                        )
                                    )
                            # or a DataZippedBlock
                            elif id_string == b"##DZ":
                                (
                                    original_type,
                                    zip_type,
                                    param,
                                    original_size,
                                    zip_size,
                                ) = v4c.DZ_COMMON_INFO_uf(stream, addr)

                                if original_size:
                                    if zip_type == v4c.FLAG_DZ_DEFLATE:
                                        block_type_ = v4c.DZ_BLOCK_DEFLATE
                                        param = 0
                                    else:
                                        block_type_ = v4c.DZ_BLOCK_TRANSPOSED
                                    if total_size < original_size:
                                        block_limit = total_size
                                    else:
                                        block_limit = None
                                    total_size -= original_size
                                    info.append(
                                        DataBlockInfo(
                                            address=addr + v4c.DZ_COMMON_SIZE,
                                            block_type=block_type_,
                                            original_size=original_size,
                                            compressed_size=zip_size,
                                            param=param,
                                            block_limit=block_limit,
                                        )
                                    )

                            if has_invalidation:
                                inval_addr = ld[f"invalidation_bits_addr_{i}"]
                                if inval_addr:
                                    id_string, block_len = COMMON_SHORT_uf(
                                        stream, inval_addr
                                    )
                                    if id_string == b"##DI":
                                        size = block_len - 24
                                        if size:
                                            if inval_total_size < size:
                                                block_limit = inval_total_size
                                            else:
                                                block_limit = None
                                            inval_total_size -= size
                                            info[
                                                -1
                                            ].invalidation_block = InvalidationBlockInfo(
                                                address=inval_addr + COMMON_SIZE,
                                                block_type=v4c.DT_BLOCK,
                                                original_size=size,
                                                compressed_size=size,
                                                param=0,
                                                block_limit=block_limit,
                                            )
                                    else:
                                        (
                                            original_type,
                                            zip_type,
                                            param,
                                            original_size,
                                            zip_size,
                                        ) = v4c.DZ_COMMON_INFO_uf(stream, inval_addr)

                                        if original_size:
                                            if zip_type == v4c.FLAG_DZ_DEFLATE:
                                                block_type_ = v4c.DZ_BLOCK_DEFLATE
                                                param = 0
                                            else:
                                                block_type_ = v4c.DZ_BLOCK_TRANSPOSED
                                            if inval_total_size < original_size:
                                                block_limit = inval_total_size
                                            else:
                                                block_limit = None
                                            inval_total_size -= original_size
                                            info[
                                                -1
                                            ].invalidation_block = InvalidationBlockInfo(
                                                address=inval_addr + v4c.DZ_COMMON_SIZE,
                                                block_type=block_type_,
                                                original_size=original_size,
                                                compressed_size=zip_size,
                                                param=param,
                                                block_limit=block_limit,
                                            )
                                else:
                                    info[-1].invalidation_block = InvalidationBlockInfo(
                                        address=0,
                                        block_type=v4c.DT_BLOCK,
                                        original_size=None,
                                        compressed_size=None,
                                        param=None,
                                        all_valid=True,
                                    )

                        address = ld.next_ld_addr

                # or a header list
                elif id_string == b"##HL":
                    hl = HeaderList(address=address, stream=stream, mapped=mapped)
                    address = hl.first_dl_addr

                    info, uses_ld = self._get_data_blocks_info(
                        address,
                        stream,
                        block_type,
                        mapped,
                        total_size,
                        inval_total_size,
                    )
        else:

            if address:
                stream.seek(address)
                id_string, block_len = COMMON_SHORT_u(stream.read(COMMON_SHORT_SIZE))

                # can be a DataBlock
                if id_string == block_type:
                    size = block_len - 24
                    if size:
                        if total_size < size:
                            block_limit = total_size
                        else:
                            block_limit = None
                        total_size -= size
                        info.append(
                            DataBlockInfo(
                                address=address + COMMON_SIZE,
                                block_type=v4c.DT_BLOCK,
                                original_size=size,
                                compressed_size=size,
                                param=0,
                                block_limit=block_limit,
                            )
                        )
                # or a DataZippedBlock
                elif id_string == b"##DZ":
                    stream.seek(address)
                    (
                        original_type,
                        zip_type,
                        param,
                        original_size,
                        zip_size,
                    ) = v4c.DZ_COMMON_INFO_u(stream.read(v4c.DZ_COMMON_SIZE))

                    if original_size:
                        if zip_type == v4c.FLAG_DZ_DEFLATE:
                            block_type_ = v4c.DZ_BLOCK_DEFLATE
                            param = 0
                        else:
                            block_type_ = v4c.DZ_BLOCK_TRANSPOSED
                        if total_size < original_size:
                            block_limit = total_size
                        else:
                            block_limit = None
                        total_size -= original_size
                        info.append(
                            DataBlockInfo(
                                address=address + v4c.DZ_COMMON_SIZE,
                                block_type=block_type_,
                                original_size=original_size,
                                compressed_size=zip_size,
                                param=param,
                                block_limit=block_limit,
                            )
                        )

                # or a DataList
                elif id_string == b"##DL":
                    while address:
                        dl = DataList(address=address, stream=stream)
                        for i in range(dl.data_block_nr):

                            addr = dl[f"data_block_addr{i}"]

                            stream.seek(addr)
                            id_string, block_len = COMMON_SHORT_u(
                                stream.read(COMMON_SHORT_SIZE)
                            )

                            # can be a DataBlock
                            if id_string == block_type:
                                size = block_len - 24
                                if size:
                                    if total_size < size:
                                        block_limit = total_size
                                    else:
                                        block_limit = None
                                    total_size -= size
                                    info.append(
                                        DataBlockInfo(
                                            address=addr + COMMON_SIZE,
                                            block_type=v4c.DT_BLOCK,
                                            original_size=size,
                                            compressed_size=size,
                                            param=0,
                                            block_limit=block_limit,
                                        )
                                    )
                            # or a DataZippedBlock
                            elif id_string == b"##DZ":
                                stream.seek(addr)
                                (
                                    original_type,
                                    zip_type,
                                    param,
                                    original_size,
                                    zip_size,
                                ) = v4c.DZ_COMMON_INFO_u(
                                    stream.read(v4c.DZ_COMMON_SIZE)
                                )

                                if original_size:
                                    if zip_type == v4c.FLAG_DZ_DEFLATE:
                                        block_type_ = v4c.DZ_BLOCK_DEFLATE
                                        param = 0
                                    else:
                                        block_type_ = v4c.DZ_BLOCK_TRANSPOSED
                                    if total_size < original_size:
                                        block_limit = total_size
                                    else:
                                        block_limit = None
                                    total_size -= original_size
                                    info.append(
                                        DataBlockInfo(
                                            address=addr + v4c.DZ_COMMON_SIZE,
                                            block_type=block_type_,
                                            original_size=original_size,
                                            compressed_size=zip_size,
                                            param=param,
                                            block_limit=block_limit,
                                        )
                                    )
                        address = dl.next_dl_addr

                # or a DataList
                elif id_string == b"##LD":
                    uses_ld = True
                    while address:
                        ld = ListData(address=address, stream=stream)
                        has_invalidation = ld.flags & v4c.FLAG_LD_INVALIDATION_PRESENT
                        for i in range(ld.data_block_nr):
                            addr = ld[f"data_block_addr{i}"]

                            stream.seek(addr)
                            id_string, block_len = COMMON_SHORT_u(
                                stream.read(COMMON_SHORT_SIZE)
                            )
                            # can be a DataBlock
                            if id_string == b"##DV":
                                size = block_len - 24
                                if size:
                                    if total_size < size:
                                        block_limit = total_size
                                    else:
                                        block_limit = None
                                    total_size -= size
                                    info.append(
                                        DataBlockInfo(
                                            address=addr + COMMON_SIZE,
                                            block_type=v4c.DT_BLOCK,
                                            original_size=size,
                                            compressed_size=size,
                                            param=0,
                                            block_limit=block_limit,
                                        )
                                    )
                            # or a DataZippedBlock
                            elif id_string == b"##DZ":
                                stream.seek(addr)
                                (
                                    original_type,
                                    zip_type,
                                    param,
                                    original_size,
                                    zip_size,
                                ) = v4c.DZ_COMMON_INFO_u(
                                    stream.read(v4c.DZ_COMMON_SIZE)
                                )

                                if original_size:
                                    if zip_type == v4c.FLAG_DZ_DEFLATE:
                                        block_type_ = v4c.DZ_BLOCK_DEFLATE
                                        param = 0
                                    else:
                                        block_type_ = v4c.DZ_BLOCK_TRANSPOSED
                                    if total_size < original_size:
                                        block_limit = total_size
                                    else:
                                        block_limit = None
                                    total_size -= original_size
                                    info.append(
                                        DataBlockInfo(
                                            address=addr + v4c.DZ_COMMON_SIZE,
                                            block_type=block_type_,
                                            original_size=original_size,
                                            compressed_size=zip_size,
                                            param=param,
                                            block_limit=block_limit,
                                        )
                                    )

                            if has_invalidation:
                                inval_addr = ld[f"invalidation_bits_addr_{i}"]
                                if inval_addr:
                                    stream.seek(inval_addr)
                                    id_string, block_len = COMMON_SHORT_u(
                                        stream.read(COMMON_SHORT_SIZE)
                                    )
                                    if id_string == b"##DI":
                                        size = block_len - 24
                                        if size:
                                            if inval_total_size < size:
                                                block_limit = inval_total_size
                                            else:
                                                block_limit = None
                                            inval_total_size -= size
                                            info[
                                                -1
                                            ].invalidation_block = InvalidationBlockInfo(
                                                address=inval_addr + COMMON_SIZE,
                                                block_type=v4c.DT_BLOCK,
                                                original_size=size,
                                                compressed_size=size,
                                                param=0,
                                                block_limit=block_limit,
                                            )
                                    else:
                                        (
                                            original_type,
                                            zip_type,
                                            param,
                                            original_size,
                                            zip_size,
                                        ) = v4c.DZ_COMMON_INFO_u(
                                            stream.read(v4c.DZ_COMMON_SIZE)
                                        )

                                        if original_size:
                                            if zip_type == v4c.FLAG_DZ_DEFLATE:
                                                block_type_ = v4c.DZ_BLOCK_DEFLATE
                                                param = 0
                                            else:
                                                block_type_ = v4c.DZ_BLOCK_TRANSPOSED
                                            if inval_total_size < original_size:
                                                block_limit = inval_total_size
                                            else:
                                                block_limit = None
                                            inval_total_size -= original_size
                                            info[
                                                -1
                                            ].invalidation_block = InvalidationBlockInfo(
                                                address=inval_addr + v4c.DZ_COMMON_SIZE,
                                                block_type=block_type_,
                                                original_size=original_size,
                                                compressed_size=zip_size,
                                                param=param,
                                                block_limit=block_limit,
                                            )
                                else:
                                    info[-1].invalidation_block = InvalidationBlockInfo(
                                        address=0,
                                        block_type=v4c.DT_BLOCK,
                                        original_size=0,
                                        compressed_size=0,
                                        param=0,
                                        all_valid=True,
                                    )
                        address = ld.next_ld_addr

                # or a header list
                elif id_string == b"##HL":
                    hl = HeaderList(address=address, stream=stream)
                    address = hl.first_dl_addr

                    info, uses_ld = self._get_data_blocks_info(
                        address,
                        stream,
                        block_type,
                        mapped,
                        total_size,
                        inval_total_size,
                    )

        return info, uses_ld

    def _get_signal_data_blocks_info(
        self,
        address,
        stream,
    ):
        info = []

        if address:
            stream.seek(address)
            id_string, block_len = COMMON_SHORT_u(stream.read(COMMON_SHORT_SIZE))

            # can be a DataBlock
            if id_string == b"##SD":
                size = block_len - 24
                if size:
                    info.append(
                        SignalDataBlockInfo(
                            address=address + COMMON_SIZE,
                            compressed_size=size,
                            original_size=size,
                            block_type=v4c.DT_BLOCK,
                        )
                    )

            # or a DataZippedBlock
            elif id_string == b"##DZ":
                stream.seek(address)
                (
                    original_type,
                    zip_type,
                    param,
                    original_size,
                    zip_size,
                ) = v4c.DZ_COMMON_INFO_u(stream.read(v4c.DZ_COMMON_SIZE))

                if original_size:
                    if zip_type == v4c.FLAG_DZ_DEFLATE:
                        block_type_ = v4c.DZ_BLOCK_DEFLATE
                        param = 0
                    else:
                        block_type_ = v4c.DZ_BLOCK_TRANSPOSED

                    info.append(
                        SignalDataBlockInfo(
                            address=address + v4c.DZ_COMMON_SIZE,
                            block_type=block_type_,
                            original_size=original_size,
                            compressed_size=zip_size,
                            param=param,
                        )
                    )

            # or a DataList
            elif id_string == b"##DL":
                while address:
                    dl = DataList(address=address, stream=stream)
                    for i in range(dl.data_block_nr):

                        addr = dl[f"data_block_addr{i}"]

                        stream.seek(addr)
                        id_string, block_len = COMMON_SHORT_u(
                            stream.read(COMMON_SHORT_SIZE)
                        )

                        # can be a DataBlock
                        if id_string == b"##SD":
                            size = block_len - 24
                            if size:
                                info.append(
                                    SignalDataBlockInfo(
                                        address=addr + COMMON_SIZE,
                                        compressed_size=size,
                                        original_size=size,
                                        block_type=v4c.DT_BLOCK,
                                    )
                                )
                        # or a DataZippedBlock
                        elif id_string == b"##DZ":
                            stream.seek(addr)
                            (
                                original_type,
                                zip_type,
                                param,
                                original_size,
                                zip_size,
                            ) = v4c.DZ_COMMON_INFO_u(stream.read(v4c.DZ_COMMON_SIZE))

                            if original_size:
                                if zip_type == v4c.FLAG_DZ_DEFLATE:
                                    block_type_ = v4c.DZ_BLOCK_DEFLATE
                                    param = 0
                                else:
                                    block_type_ = v4c.DZ_BLOCK_TRANSPOSED
                                info.append(
                                    SignalDataBlockInfo(
                                        address=addr + v4c.DZ_COMMON_SIZE,
                                        block_type=block_type_,
                                        original_size=original_size,
                                        compressed_size=zip_size,
                                        param=param,
                                    )
                                )
                    address = dl.next_dl_addr

            # or a header list
            elif id_string == b"##HL":
                hl = HeaderList(address=address, stream=stream)
                address = hl.first_dl_addr

                info = self._get_signal_data_blocks_info(
                    address,
                    stream,
                )

        return info or None

    def _filter_occurrences(
        self, occurrences, source_name=None, source_path=None, acq_name=None
    ):
        if source_name is not None:
            occurrences = (
                (gp_idx, cn_idx)
                for gp_idx, cn_idx in occurrences
                if (
                    self.groups[gp_idx].channels[cn_idx].source is not None
                    and self.groups[gp_idx].channels[cn_idx].source.name == source_name
                )
                or (
                    self.groups[gp_idx].channel_group.acq_source is not None
                    and self.groups[gp_idx].channel_group.acq_source.name == source_name
                )
            )

        if source_path is not None:
            occurrences = (
                (gp_idx, cn_idx)
                for gp_idx, cn_idx in occurrences
                if (
                    self.groups[gp_idx].channels[cn_idx].source is not None
                    and self.groups[gp_idx].channels[cn_idx].source.path == source_path
                )
                or (
                    self.groups[gp_idx].channel_group.acq_source is not None
                    and self.groups[gp_idx].channel_group.acq_source.path == source_path
                )
            )

        if acq_name is not None:
            occurrences = (
                (gp_idx, cn_idx)
                for gp_idx, cn_idx in occurrences
                if self.groups[gp_idx].channel_group.acq_name == acq_name
            )

        return occurrences

    def get_invalidation_bits(self, group_index, channel, fragment):
        """get invalidation indexes for the channel

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
        invalidation_bits : iterable
            iterable of valid channel indexes; if all are valid `None` is
            returned

        """
        group = self.groups[group_index]
        dtypes = group.types

        data_bytes, offset, _count, invalidation_bytes = fragment
        try:
            invalidation = self._invalidation_cache[(group_index, offset, _count)]
        except KeyError:
            if invalidation_bytes is not None:
                size = group.channel_group.invalidation_bytes_nr
                invalidation = frombuffer(invalidation_bytes, dtype=f"({size},)u1")

            else:
                record = group.record
                if record is None:
                    dtypes = group.types
                    if dtypes.itemsize:
                        record = fromstring(data_bytes, dtype=dtypes)
                    else:
                        record = None

                invalidation = record["invalidation_bytes"].copy()
            self._invalidation_cache[(group_index, offset, _count)] = invalidation

        ch_invalidation_pos = channel.pos_invalidation_bit
        pos_byte, pos_offset = ch_invalidation_pos // 8, ch_invalidation_pos % 8

        mask = 1 << pos_offset

        invalidation_bits = invalidation[:, pos_byte] & mask
        invalidation_bits = invalidation_bits.astype(bool)

        return invalidation_bits

    def configure(
        self,
        *,
        from_other=None,
        read_fragment_size=None,
        write_fragment_size=None,
        use_display_names=None,
        single_bit_uint_as_bool=None,
        integer_interpolation=None,
        copy_on_get=None,
        float_interpolation=None,
        raise_on_multiple_occurrences=None,
    ):
        """configure MDF parameters.

        The default values for the options are the following:
        * read_fragment_size = 0
        * write_fragment_size = 4MB
        * use_display_names = False
        * single_bit_uint_as_bool = False
        * integer_interpolation = 0 (ffill - use previous sample)
        * float_interpolation = 1 (linear interpolation)
        * copy_on_get = False
        * raise_on_multiple_occurrences = True

        Parameters
        ----------
        read_fragment_size : int
            size hint of split data blocks, default 8MB; if the initial size is
            smaller, then no data list is used. The actual split size depends on
            the data groups' records size
        write_fragment_size : int
            size hint of split data blocks, default 4MB; if the initial size is
            smaller, then no data list is used. The actual split size depends on
            the data groups' records size. Maximum size is 4MB to ensure
            compatibility with CANape
        use_display_names : bool
            search for display name in the Channel XML comment
        single_bit_uint_as_bool : bool
            return single bit channels are np.bool arrays
        integer_interpolation : int
            interpolation mode for integer channels:

                * 0 - repeat previous sample
                * 1 - use linear interpolation
                * 2 - hybrid interpolation: channels with integer data type (raw values) that have a
                  conversion that outputs float values will use linear interpolation, otherwise
                  the previous sample is used

                .. versionchanged:: 6.2.0
                    added hybrid mode interpolation

        copy_on_get : bool
            copy arrays in the get method

        float_interpolation : int
            interpolation mode for float channels:

                * 0 - repeat previous sample
                * 1 - use linear interpolation

                .. versionadded:: 6.2.0

        raise_on_multiple_occurrences : bool
            raise exception when there are multiple channel occurrences in the file and
            the `get` call is ambiguous; default True

            .. versionadded:: 6.2.0

        from_other : MDF
            copy configuration options from other MDF

            .. versionadded:: 6.2.0

        """

        if from_other is not None:
            self._read_fragment_size = from_other._read_fragment_size
            self._write_fragment_size = from_other._write_fragment_size
            self._use_display_names = from_other._use_display_names
            self._single_bit_uint_as_bool = from_other._single_bit_uint_as_bool
            self._integer_interpolation = from_other._integer_interpolation
            self.copy_on_get = from_other.copy_on_get
            self._float_interpolation = from_other._float_interpolation
            self._raise_on_multiple_occurrences = (
                from_other._raise_on_multiple_occurrences
            )

        if read_fragment_size is not None:
            self._read_fragment_size = int(read_fragment_size)

        if write_fragment_size:
            self._write_fragment_size = min(int(write_fragment_size), 4 * 2 ** 20)

        if use_display_names is not None:
            self._use_display_names = bool(use_display_names)

        if single_bit_uint_as_bool is not None:
            self._single_bit_uint_as_bool = bool(single_bit_uint_as_bool)

        if integer_interpolation in (0, 1, 2):
            self._integer_interpolation = int(integer_interpolation)

        if copy_on_get is not None:
            self.copy_on_get = copy_on_get

        if float_interpolation in (0, 1):
            self._float_interpolation = int(float_interpolation)

        if raise_on_multiple_occurrences is not None:
            self._raise_on_multiple_occurrences = bool(raise_on_multiple_occurrences)

    def append(
        self,
        signals,
        acq_name=None,
        acq_source=None,
        comment="Python",
        common_timebase=False,
        units=None,
    ):
        """
        Appends a new data group.

        For channel dependencies type Signals, the *samples* attribute must be
        a numpy.recarray

        Parameters
        ----------
        signals : list | Signal | pandas.DataFrame
            list of *Signal* objects, or a single *Signal* object, or a pandas
            *DataFrame* object. All bytes columns in the pandas *DataFrame*
            must be *utf-8* encoded
        acq_name : str
            channel group acquisition name
        acq_source : Source
            channel group acquisition source
        comment : str
            channel group comment; default 'Python'
        common_timebase : bool
            flag to hint that the signals have the same timebase. Only set this
            if you know for sure that all appended channels share the same
            time base
        units : dict
            will contain the signal units mapped to the signal names when
            appending a pandas DataFrame

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
        >>> s1 = Signal(samples=s1, timestamps=t, unit='+', name='Positive')
        >>> s2 = Signal(samples=s2, timestamps=t, unit='-', name='Negative')
        >>> s3 = Signal(samples=s3, timestamps=t, unit='flts', name='Floats')
        >>> mdf = MDF4('new.mdf')
        >>> mdf.append([s1, s2, s3], comment='created by asammdf v4.0.0')
        >>> # case 2: VTAB conversions from channels inside another file
        >>> mdf1 = MDF4('in.mf4')
        >>> ch1 = mdf1.get("Channel1_VTAB")
        >>> ch2 = mdf1.get("Channel2_VTABR")
        >>> sigs = [ch1, ch2]
        >>> mdf2 = MDF4('out.mf4')
        >>> mdf2.append(sigs, comment='created by asammdf v4.0.0')
        >>> mdf2.append(ch1, comment='just a single channel')
        >>> df = pd.DataFrame.from_dict({'s1': np.array([1, 2, 3, 4, 5]), 's2': np.array([-1, -2, -3, -4, -5])})
        >>> units = {'s1': 'V', 's2': 'A'}
        >>> mdf2.append(df, units=units)

        """
        source_block = (
            SourceInformation.from_common_source(acq_source)
            if acq_source
            else acq_source
        )

        if isinstance(signals, Signal):
            signals = [signals]
        elif isinstance(signals, DataFrame):
            self._append_dataframe(
                signals,
                acq_name=acq_name,
                acq_source=source_block,
                comment=comment,
                units=units,
            )
            return

        if not signals:
            return

        prepare_record = True

        # check if the signals have a common timebase
        # if not interpolate the signals using the union of all timebases
        if signals:
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
                    t = unique(concatenate(times)).astype(float64)
                    signals = [
                        s.interp(
                            t,
                            integer_interpolation_mode=self._integer_interpolation,
                            float_interpolation_mode=self._float_interpolation,
                        )
                        for s in signals
                    ]
                else:
                    t = t_
            else:
                t = t_
        else:
            t = []

        if self.version >= "4.20" and (self._column_storage or 1):
            return self._append_column_oriented(
                signals, acq_name=acq_name, acq_source=source_block, comment=comment
            )

        dg_cntr = len(self.groups)

        gp = Group(None)
        gp.signal_data = gp_sdata = []
        gp.channels = gp_channels = []
        gp.channel_dependencies = gp_dep = []
        gp.signal_types = gp_sig_types = []

        cycles_nr = len(t)

        # channel group
        kwargs = {"cycles_nr": cycles_nr, "samples_byte_nr": 0}
        gp.channel_group = ChannelGroup(**kwargs)
        gp.channel_group.acq_name = acq_name
        gp.channel_group.acq_source = source_block
        gp.channel_group.comment = comment

        if any(sig.invalidation_bits is not None for sig in signals):
            invalidation_bytes_nr = 1
            gp.channel_group.invalidation_bytes_nr = invalidation_bytes_nr

            inval_bits = []

        else:
            invalidation_bytes_nr = 0
            inval_bits = []
        inval_cntr = 0

        self.groups.append(gp)

        fields = []
        types = []
        parents = {}
        ch_cntr = 0
        offset = 0
        field_names = UniqueDB()

        defined_texts = {}
        si_map = self._si_map

        # setup all blocks related to the time master channel

        file = self._tempfile
        tell = file.tell
        seek = file.seek

        seek(0, 2)

        if signals:
            master_metadata = signals[0].master_metadata
        else:
            master_metadata = None
        if master_metadata:
            time_name, sync_type = master_metadata
            if sync_type in (0, 1):
                time_unit = "s"
            elif sync_type == 2:
                time_unit = "deg"
            elif sync_type == 3:
                time_unit = "m"
            elif sync_type == 4:
                time_unit = "index"
        else:
            time_name, sync_type = "time", v4c.SYNC_TYPE_TIME
            time_unit = "s"

        gp.channel_group.acq_source = source_block

        if signals:

            # time channel
            t_type, t_size = fmt_to_datatype_v4(t.dtype, t.shape)
            kwargs = {
                "channel_type": v4c.CHANNEL_TYPE_MASTER,
                "data_type": t_type,
                "sync_type": sync_type,
                "byte_offset": 0,
                "bit_offset": 0,
                "bit_count": t_size,
            }

            ch = Channel(**kwargs)
            ch.unit = time_unit
            ch.name = time_name
            ch.source = source_block
            ch.dtype_fmt = t.dtype
            name = time_name

            gp_channels.append(ch)

            gp_sdata.append(None)
            self.channels_db.add(name, (dg_cntr, ch_cntr))
            self.masters_db[dg_cntr] = 0
            # data group record parents
            parents[ch_cntr] = name, 0

            # time channel doesn't have channel dependencies
            gp_dep.append(None)

            fields.append(t)
            types.append((name, t.dtype))
            field_names.get_unique_name(name)

            offset += t_size // 8
            ch_cntr += 1

            gp_sig_types.append(0)

        for signal in signals:
            sig = signal
            samples = sig.samples
            sig_dtype = samples.dtype
            sig_shape = samples.shape
            names = sig_dtype.names
            name = signal.name

            if names is None:
                sig_type = v4c.SIGNAL_TYPE_SCALAR
                if sig_dtype.kind in "SV":
                    sig_type = v4c.SIGNAL_TYPE_STRING
            else:
                prepare_record = False
                if names in (v4c.CANOPEN_TIME_FIELDS, v4c.CANOPEN_DATE_FIELDS):
                    sig_type = v4c.SIGNAL_TYPE_CANOPEN
                elif names[0] != sig.name:
                    sig_type = v4c.SIGNAL_TYPE_STRUCTURE_COMPOSITION
                else:
                    sig_type = v4c.SIGNAL_TYPE_ARRAY

            gp_sig_types.append(sig_type)

            # first add the signals in the simple signal list
            if sig_type == v4c.SIGNAL_TYPE_SCALAR:

                # compute additional byte offset for large records size
                s_type, s_size = fmt_to_datatype_v4(sig_dtype, sig_shape)

                byte_size = s_size // 8 or 1
                data_block_addr = 0

                if sig_dtype.kind == "u" and signal.bit_count <= 4:
                    s_size = signal.bit_count

                if signal.stream_sync:
                    channel_type = v4c.CHANNEL_TYPE_SYNC
                    if signal.attachment:
                        at_data, at_name, hash_sum = signal.attachment
                        attachment_index = self.attach(
                            at_data, at_name, hash_sum, mime="video/avi", embedded=False
                        )
                        attachment = attachment_index
                    else:
                        attachment = None

                    sync_type = v4c.SYNC_TYPE_TIME
                else:
                    channel_type = v4c.CHANNEL_TYPE_VALUE
                    sync_type = v4c.SYNC_TYPE_NONE

                    if signal.attachment:
                        at_data, at_name, hash_sum = signal.attachment

                        attachment_index = self.attach(at_data, at_name, hash_sum)
                        attachment = attachment_index
                    else:
                        attachment = None

                kwargs = {
                    "channel_type": channel_type,
                    "sync_type": sync_type,
                    "bit_count": s_size,
                    "byte_offset": offset,
                    "bit_offset": 0,
                    "data_type": s_type,
                    "data_block_addr": data_block_addr,
                    "flags": 0,
                }

                if attachment is not None:
                    kwargs["attachment_addr"] = 0

                if invalidation_bytes_nr and signal.invalidation_bits is not None:
                    inval_bits.append(signal.invalidation_bits)
                    kwargs["flags"] = v4c.FLAG_CN_INVALIDATION_PRESENT
                    kwargs["pos_invalidation_bit"] = inval_cntr
                    inval_cntr += 1

                ch = Channel(**kwargs)
                ch.name = name
                ch.unit = signal.unit
                ch.comment = signal.comment
                ch.display_name = signal.display_name
                if len(sig_shape) > 1:
                    ch.dtype_fmt = dtype((sig_dtype, sig_shape[1:]))
                else:
                    ch.dtype_fmt = sig_dtype
                ch.attachment = attachment

                # conversions for channel
                if signal.raw:
                    ch.conversion = conversion_transfer(signal.conversion, version=4)

                # source for channel

                source = signal.source
                if source:
                    if source in si_map:
                        ch.source = si_map[source]
                    else:
                        new_source = SourceInformation(
                            source_type=source.source_type, bus_type=source.bus_type
                        )
                        new_source.name = source.name
                        new_source.path = source.path
                        new_source.comment = source.comment

                        si_map[source] = new_source

                        ch.source = new_source

                gp_channels.append(ch)

                offset += byte_size

                gp_sdata.append(None)
                entry = (dg_cntr, ch_cntr)
                self.channels_db.add(name, entry)
                if ch.display_name:
                    self.channels_db.add(ch.display_name, entry)

                # update the parents as well
                field_name = field_names.get_unique_name(name)
                parents[ch_cntr] = field_name, 0

                fields.append(samples)
                types.append((field_name, sig_dtype, sig_shape[1:]))

                ch_cntr += 1

                # simple channels don't have channel dependencies
                gp_dep.append(None)

            elif sig_type == v4c.SIGNAL_TYPE_CANOPEN:

                field_name = field_names.get_unique_name(name)

                if names == v4c.CANOPEN_TIME_FIELDS:

                    vals = signal.samples.tobytes()

                    fields.append(frombuffer(vals, dtype="V6"))
                    types.append((field_name, "V6"))
                    byte_size = 6
                    s_type = v4c.DATA_TYPE_CANOPEN_TIME
                    s_dtype = dtype("V6")

                else:
                    vals = []
                    for field in ("ms", "min", "hour", "day", "month", "year"):
                        if field == "hour":
                            vals.append(
                                signal.samples[field]
                                + (signal.samples["summer_time"] << 7)
                            )
                        elif field == "day":
                            vals.append(
                                signal.samples[field]
                                + (signal.samples["day_of_week"] << 4)
                            )
                        else:
                            vals.append(signal.samples[field])
                    vals = fromarrays(vals).tobytes()

                    fields.append(frombuffer(vals, dtype="V7"))
                    types.append((field_name, "V7"))
                    byte_size = 7
                    s_type = v4c.DATA_TYPE_CANOPEN_DATE
                    s_dtype = dtype("V7")

                s_size = byte_size * 8

                # there is no channel dependency
                gp_dep.append(None)

                # add channel block
                kwargs = {
                    "channel_type": v4c.CHANNEL_TYPE_VALUE,
                    "bit_count": s_size,
                    "byte_offset": offset,
                    "bit_offset": 0,
                    "data_type": s_type,
                    "flags": 0,
                }
                if invalidation_bytes_nr and signal.invalidation_bits is not None:
                    inval_bits.append(signal.invalidation_bits)
                    kwargs["flags"] |= v4c.FLAG_CN_INVALIDATION_PRESENT
                    kwargs["pos_invalidation_bit"] = inval_cntr
                    inval_cntr += 1

                ch = Channel(**kwargs)
                ch.name = name
                ch.unit = signal.unit
                ch.comment = signal.comment
                ch.display_name = signal.display_name
                ch.dtype_fmt = s_dtype

                # source for channel
                source = signal.source
                if source:
                    if source in si_map:
                        ch.source = si_map[source]
                    else:
                        new_source = SourceInformation(
                            source_type=source.source_type, bus_type=source.bus_type
                        )
                        new_source.name = source.name
                        new_source.path = source.path
                        new_source.comment = source.comment

                        si_map[source] = new_source

                        ch.source = new_source

                gp_channels.append(ch)

                offset += byte_size

                entry = (dg_cntr, ch_cntr)
                self.channels_db.add(name, entry)
                if ch.display_name:
                    self.channels_db.add(ch.display_name, entry)

                # update the parents as well
                parents[ch_cntr] = field_name, 0

                gp_sdata.append(0)

                ch_cntr += 1

            elif sig_type == v4c.SIGNAL_TYPE_STRUCTURE_COMPOSITION:
                (
                    offset,
                    dg_cntr,
                    ch_cntr,
                    struct_self,
                    new_fields,
                    new_types,
                    inval_cntr,
                ) = self._append_structure_composition(
                    gp,
                    signal,
                    field_names,
                    offset,
                    dg_cntr,
                    ch_cntr,
                    parents,
                    defined_texts,
                    invalidation_bytes_nr,
                    inval_bits,
                    inval_cntr,
                )
                fields.extend(new_fields)
                types.extend(new_types)

            elif sig_type == v4c.SIGNAL_TYPE_ARRAY:
                # here we have channel arrays or mdf v3 channel dependencies
                samples = signal.samples[names[0]]
                shape = samples.shape[1:]

                if len(names) > 1 or len(shape) > 1:
                    # add channel dependency block for composed parent channel
                    dims_nr = len(shape)
                    names_nr = len(names)

                    if names_nr == 0:
                        kwargs = {
                            "dims": dims_nr,
                            "ca_type": v4c.CA_TYPE_LOOKUP,
                            "flags": v4c.FLAG_CA_FIXED_AXIS,
                            "byte_offset_base": samples.dtype.itemsize,
                        }
                        for i in range(dims_nr):
                            kwargs[f"dim_size_{i}"] = shape[i]

                    elif len(names) == 1:
                        kwargs = {
                            "dims": dims_nr,
                            "ca_type": v4c.CA_TYPE_ARRAY,
                            "flags": 0,
                            "byte_offset_base": samples.dtype.itemsize,
                        }
                        for i in range(dims_nr):
                            kwargs[f"dim_size_{i}"] = shape[i]

                    else:
                        kwargs = {
                            "dims": dims_nr,
                            "ca_type": v4c.CA_TYPE_LOOKUP,
                            "flags": v4c.FLAG_CA_AXIS,
                            "byte_offset_base": samples.dtype.itemsize,
                        }
                        for i in range(dims_nr):
                            kwargs[f"dim_size_{i}"] = shape[i]

                    parent_dep = ChannelArrayBlock(**kwargs)
                    gp_dep.append([parent_dep])

                else:
                    # add channel dependency block for composed parent channel
                    kwargs = {
                        "dims": 1,
                        "ca_type": v4c.CA_TYPE_ARRAY,
                        "flags": 0,
                        "byte_offset_base": samples.dtype.itemsize,
                        "dim_size_0": shape[0],
                    }
                    parent_dep = ChannelArrayBlock(**kwargs)
                    gp_dep.append([parent_dep])

                field_name = field_names.get_unique_name(name)

                fields.append(samples)
                dtype_pair = field_name, samples.dtype, shape
                types.append(dtype_pair)

                # first we add the structure channel
                s_type, s_size = fmt_to_datatype_v4(samples.dtype, samples.shape, True)

                # add channel block
                kwargs = {
                    "channel_type": v4c.CHANNEL_TYPE_VALUE,
                    "bit_count": s_size,
                    "byte_offset": offset,
                    "bit_offset": 0,
                    "data_type": s_type,
                    "flags": 0,
                }

                if invalidation_bytes_nr:
                    if signal.invalidation_bits is not None:
                        inval_bits.append(signal.invalidation_bits)
                        kwargs["flags"] |= v4c.FLAG_CN_INVALIDATION_PRESENT
                        kwargs["pos_invalidation_bit"] = inval_cntr
                        inval_cntr += 1

                ch = Channel(**kwargs)
                ch.name = name
                ch.unit = signal.unit
                ch.comment = signal.comment
                ch.display_name = signal.display_name
                ch.dtype_fmt = samples.dtype

                # source for channel
                source = signal.source
                if source:
                    if source in si_map:
                        ch.source = si_map[source]
                    else:
                        new_source = SourceInformation(
                            source_type=source.source_type, bus_type=source.bus_type
                        )
                        new_source.name = source.name
                        new_source.path = source.path
                        new_source.comment = source.comment

                        si_map[source] = new_source

                        ch.source = new_source

                gp_channels.append(ch)

                size = s_size // 8
                for dim in shape:
                    size *= dim
                offset += size

                gp_sdata.append(None)
                entry = (dg_cntr, ch_cntr)
                self.channels_db.add(name, entry)
                if ch.display_name:
                    self.channels_db.add(ch.display_name, entry)

                # update the parents as well
                parents[ch_cntr] = name, 0

                ch_cntr += 1

                for name in names[1:]:
                    field_name = field_names.get_unique_name(name)

                    samples = signal.samples[name]
                    shape = samples.shape[1:]
                    fields.append(samples)
                    types.append((field_name, samples.dtype, shape))

                    # add channel dependency block
                    kwargs = {
                        "dims": 1,
                        "ca_type": v4c.CA_TYPE_SCALE_AXIS,
                        "flags": 0,
                        "byte_offset_base": samples.dtype.itemsize,
                        "dim_size_0": shape[0],
                    }
                    dep = ChannelArrayBlock(**kwargs)
                    gp_dep.append([dep])

                    # add components channel
                    s_type, s_size = fmt_to_datatype_v4(samples.dtype, ())
                    byte_size = s_size // 8 or 1
                    kwargs = {
                        "channel_type": v4c.CHANNEL_TYPE_VALUE,
                        "bit_count": s_size,
                        "byte_offset": offset,
                        "bit_offset": 0,
                        "data_type": s_type,
                        "flags": 0,
                    }

                    if invalidation_bytes_nr:
                        if signal.invalidation_bits is not None:
                            inval_bits.append(signal.invalidation_bits)
                            kwargs["flags"] |= v4c.FLAG_CN_INVALIDATION_PRESENT
                            kwargs["pos_invalidation_bit"] = inval_cntr
                            inval_cntr += 1

                    ch = Channel(**kwargs)
                    ch.name = name
                    ch.unit = signal.unit
                    ch.comment = signal.comment
                    ch.display_name = signal.display_name
                    ch.dtype_fmt = samples.dtype

                    gp_channels.append(ch)

                    entry = dg_cntr, ch_cntr
                    parent_dep.axis_channels.append(entry)
                    for dim in shape:
                        byte_size *= dim
                    offset += byte_size

                    gp_sdata.append(None)
                    self.channels_db.add(name, entry)

                    # update the parents as well
                    parents[ch_cntr] = field_name, 0

                    ch_cntr += 1

            else:

                encoding = signal.encoding
                samples = signal.samples
                sig_dtype = samples.dtype

                if encoding == "utf-8":
                    data_type = v4c.DATA_TYPE_STRING_UTF_8
                elif encoding == "latin-1":
                    data_type = v4c.DATA_TYPE_STRING_LATIN_1
                elif encoding == "utf-16-be":
                    data_type = v4c.DATA_TYPE_STRING_UTF_16_BE
                elif encoding == "utf-16-le":
                    data_type = v4c.DATA_TYPE_STRING_UTF_16_LE
                else:
                    raise MdfException(f'wrong encoding "{encoding}" for string signal')

                if self.compact_vlsd:
                    data = []
                    offsets = []
                    off = 0
                    if encoding == "utf-16-le":
                        for elem in samples:
                            offsets.append(off)
                            size = len(elem)
                            if size % 2:
                                size += 1
                                elem = elem + b"\0"
                            data.append(UINT32_p(size))
                            data.append(elem)
                            off += size + 4
                    else:
                        for elem in samples:
                            offsets.append(off)
                            size = len(elem)
                            data.append(UINT32_p(size))
                            data.append(elem)
                            off += size + 4

                    data_size = off
                    offsets = array(offsets, dtype=uint64)
                    if data_size:
                        data_addr = tell()
                        info = SignalDataBlockInfo(
                            address=data_addr,
                            compressed_size=data_size,
                            original_size=data_size,
                            location=v4c.LOCATION_TEMPORARY_FILE,
                        )
                        gp_sdata.append([info])
                        file.seek(0, 2)
                        file.write(b"".join(data))
                    else:
                        data_addr = 0
                        gp_sdata.append([])
                else:

                    offsets = arange(len(samples), dtype=uint64) * (
                        signal.samples.itemsize + 4
                    )

                    values = [
                        full(len(samples), samples.itemsize, dtype=uint32),
                        samples,
                    ]

                    types_ = [("o", uint32), ("s", sig_dtype)]

                    data = fromarrays(values, dtype=types_)

                    data_size = len(data) * data.itemsize
                    if data_size:
                        data_addr = tell()
                        info = SignalDataBlockInfo(
                            address=data_addr,
                            compressed_size=data_size,
                            original_size=data_size,
                            location=v4c.LOCATION_TEMPORARY_FILE,
                        )
                        gp_sdata.append([info])
                        data.tofile(file)
                    else:
                        data_addr = 0
                        gp_sdata.append([])

                # compute additional byte offset for large records size
                byte_size = 8
                kwargs = {
                    "channel_type": v4c.CHANNEL_TYPE_VLSD,
                    "bit_count": 64,
                    "byte_offset": offset,
                    "bit_offset": 0,
                    "data_type": data_type,
                    "data_block_addr": data_addr,
                    "flags": 0,
                }

                if invalidation_bytes_nr:
                    if signal.invalidation_bits is not None:
                        inval_bits.append(signal.invalidation_bits)
                        kwargs["flags"] |= v4c.FLAG_CN_INVALIDATION_PRESENT
                        kwargs["pos_invalidation_bit"] = inval_cntr
                        inval_cntr += 1

                ch = Channel(**kwargs)
                ch.name = name
                ch.unit = signal.unit
                ch.comment = signal.comment
                ch.display_name = signal.display_name
                ch.dtype_fmt = dtype("<u8")

                # conversions for channel
                conversion = conversion_transfer(signal.conversion, version=4)
                if signal.raw:
                    ch.conversion = conversion

                # source for channel
                source = signal.source
                if source:
                    if source in si_map:
                        ch.source = si_map[source]
                    else:
                        new_source = SourceInformation(
                            source_type=source.source_type, bus_type=source.bus_type
                        )
                        new_source.name = source.name
                        new_source.path = source.path
                        new_source.comment = source.comment

                        si_map[source] = new_source

                        ch.source = new_source

                gp_channels.append(ch)

                offset += byte_size

                entry = (dg_cntr, ch_cntr)
                self.channels_db.add(name, entry)
                if ch.display_name:
                    self.channels_db.add(ch.display_name, entry)

                # update the parents as well
                field_name = field_names.get_unique_name(name)
                parents[ch_cntr] = field_name, 0

                fields.append(offsets)
                types.append((field_name, uint64))

                ch_cntr += 1

                # simple channels don't have channel dependencies
                gp_dep.append(None)

        if invalidation_bytes_nr:
            invalidation_bytes_nr = len(inval_bits)

            for _ in range(8 - invalidation_bytes_nr % 8):
                inval_bits.append(zeros(cycles_nr, dtype=bool))

            inval_bits.reverse()
            invalidation_bytes_nr = len(inval_bits) // 8

            gp.channel_group.invalidation_bytes_nr = invalidation_bytes_nr

            inval_bits = fliplr(
                packbits(array(inval_bits).T).reshape(
                    (cycles_nr, invalidation_bytes_nr)
                )
            )

            if self.version < "4.20":

                fields.append(inval_bits)
                types.append(
                    ("invalidation_bytes", inval_bits.dtype, inval_bits.shape[1:])
                )

        gp.channel_group.cycles_nr = cycles_nr
        gp.channel_group.samples_byte_nr = offset

        virtual_group = VirtualChannelGroup()
        self.virtual_groups[dg_cntr] = virtual_group
        self.virtual_groups_map[dg_cntr] = dg_cntr
        virtual_group.groups.append(dg_cntr)
        virtual_group.record_size = offset + invalidation_bytes_nr
        virtual_group.cycles_nr = cycles_nr

        # data group
        gp.data_group = DataGroup()

        # data block
        types = dtype(types)

        gp.sorted = True
        if prepare_record:
            gp.types = types
            gp.parents = parents

        if signals and cycles_nr:
            samples = fromarrays(fields, dtype=types)
        else:
            samples = array([])

        del signals
        del fields

        size = len(samples) * samples.itemsize

        if size:
            if self.version < "4.20":

                block_size = self._write_fragment_size or 20 * 1024 * 1024

                chunk = ceil(block_size / samples.itemsize)
                count = ceil(len(samples) / chunk)

                for i in range(count):
                    data_ = samples[i * chunk : (i + 1) * chunk].tobytes()
                    raw_size = len(data_)
                    data_ = lz_compress(data_)

                    size = len(data_)
                    data_address = self._tempfile.tell()
                    self._tempfile.write(data_)

                    gp.data_blocks.append(
                        DataBlockInfo(
                            address=data_address,
                            block_type=v4c.DZ_BLOCK_LZ,
                            original_size=raw_size,
                            compressed_size=size,
                            param=0,
                        )
                    )

            else:
                data_address = self._tempfile.tell()
                gp.uses_ld = True
                data_address = tell()

                data = samples.tobytes()
                del samples
                raw_size = len(data)
                data = lz_compress(data)

                size = len(data)
                self._tempfile.write(data)

                gp.data_blocks.append(
                    DataBlockInfo(
                        address=data_address,
                        block_type=v4c.DZ_BLOCK_LZ,
                        original_size=raw_size,
                        compressed_size=size,
                        param=0,
                    )
                )

                if inval_bits is not None:
                    addr = tell()
                    data = inval_bits.tobytes()
                    raw_size = len(data)
                    data = lz_compress(data)
                    size = len(data)
                    self._tempfile.write(data)

                    gp.data_blocks[-1].invalidation_block(
                        InvalidationBlockInfo(
                            address=addr,
                            block_type=v4c.DZ_BLOCK_LZ,
                            original_size=raw_size,
                            compressed_size=size,
                            param=None,
                        )
                    )

        gp.data_location = v4c.LOCATION_TEMPORARY_FILE

        return dg_cntr

    def _append_column_oriented(
        self, signals, acq_name=None, acq_source=None, comment=None
    ):
        defined_texts = {}
        si_map = self._si_map

        # setup all blocks related to the time master channel

        file = self._tempfile
        tell = file.tell
        seek = file.seek
        write = file.write

        seek(0, 2)

        dg_cntr = initial_dg_cntr = len(self.groups)

        # add the master group

        gp = Group(None)
        gp.signal_data = gp_sdata = []
        gp.channels = gp_channels = []
        gp.channel_dependencies = gp_dep = []
        gp.signal_types = gp_sig_types = []
        gp.uses_ld = True
        gp.data_group = DataGroup()
        gp.sorted = True

        samples = signals[0].timestamps

        cycles_nr = len(samples)

        # channel group
        kwargs = {"cycles_nr": cycles_nr, "samples_byte_nr": 0}
        gp.channel_group = remote_master_channel_group = ChannelGroup(**kwargs)
        gp.channel_group.acq_name = acq_name
        gp.channel_group.acq_source = acq_source
        gp.channel_group.comment = comment

        self.groups.append(gp)

        ch_cntr = 0
        types = []
        parents = {}
        ch_cntr = 0
        offset = 0

        prepare_record = True
        source_block = None

        master_metadata = signals[0].master_metadata
        if master_metadata:
            time_name, sync_type = master_metadata
            if sync_type in (0, 1):
                time_unit = "s"
            elif sync_type == 2:
                time_unit = "deg"
            elif sync_type == 3:
                time_unit = "m"
            elif sync_type == 4:
                time_unit = "index"
        else:
            time_name, sync_type = "time", v4c.SYNC_TYPE_TIME
            time_unit = "s"

        gp.channel_group.acq_source = source_block
        # time channel
        t_type, t_size = fmt_to_datatype_v4(samples.dtype, samples.shape)
        kwargs = {
            "channel_type": v4c.CHANNEL_TYPE_MASTER,
            "data_type": t_type,
            "sync_type": sync_type,
            "byte_offset": 0,
            "bit_offset": 0,
            "bit_count": t_size,
        }

        ch = Channel(**kwargs)
        ch.unit = time_unit
        ch.name = time_name
        ch.source = source_block
        ch.dtype_fmt = samples.dtype
        name = time_name

        gp_channels.append(ch)

        gp_sdata.append(None)
        self.channels_db.add(name, (dg_cntr, ch_cntr))
        self.masters_db[dg_cntr] = 0
        # data group record parents
        parents[ch_cntr] = name, 0

        # time channel doesn't have channel dependencies
        gp_dep.append(None)

        types.append((name, samples.dtype))

        offset += t_size // 8
        ch_cntr += 1

        gp_sig_types.append(0)

        gp.channel_group.samples_byte_nr = offset

        # data group
        gp.data_group = DataGroup()

        # data block
        types = dtype(types)

        gp.sorted = True
        gp.types = types
        gp.parents = parents

        size = cycles_nr * samples.itemsize

        cg_master_index = dg_cntr

        virtual_group = VirtualChannelGroup()
        self.virtual_groups[cg_master_index] = virtual_group
        self.virtual_groups_map[dg_cntr] = dg_cntr
        virtual_group.groups.append(dg_cntr)
        virtual_group.record_size = offset
        virtual_group.cycles_nr = cycles_nr

        dg_cntr += 1

        if size:
            data_address = tell()
            gp.data_location = v4c.LOCATION_TEMPORARY_FILE
            write(samples.tobytes())

            chunk = self._write_fragment_size // samples.itemsize
            chunk *= samples.itemsize

            while size:

                if size > chunk:
                    gp.data_blocks.append(
                        DataBlockInfo(
                            address=data_address,
                            block_type=v4c.DT_BLOCK,
                            original_size=chunk,
                            compressed_size=chunk,
                            param=0,
                        )
                    )
                    data_address += chunk
                    size -= chunk
                else:
                    gp.data_blocks.append(
                        DataBlockInfo(
                            address=data_address,
                            block_type=v4c.DT_BLOCK,
                            original_size=size,
                            compressed_size=size,
                            param=0,
                        )
                    )
                    size = 0
        else:
            gp.data_location = v4c.LOCATION_TEMPORARY_FILE

        for signal in signals:
            gp = Group(None)
            gp.signal_data = gp_sdata = []
            gp.channels = gp_channels = []
            gp.channel_dependencies = gp_dep = []
            gp.signal_types = gp_sig_types = []
            gp.data_group = DataGroup()
            gp.sorted = True
            gp.uses_ld = True

            # channel group
            kwargs = {
                "cycles_nr": cycles_nr,
                "samples_byte_nr": 0,
                "flags": v4c.FLAG_CG_REMOTE_MASTER,
            }
            gp.channel_group = ChannelGroup(**kwargs)
            gp.channel_group.acq_name = acq_name
            gp.channel_group.acq_source = acq_source
            gp.channel_group.comment = remote_master_channel_group.comment
            gp.channel_group.cg_master_index = cg_master_index

            self.groups.append(gp)

            types = []
            parents = {}
            ch_cntr = 0
            offset = 0
            field_names = UniqueDB()

            sig = signal
            samples = sig.samples
            sig_dtype = samples.dtype
            sig_shape = samples.shape
            names = sig_dtype.names
            name = signal.name

            if names is None:
                sig_type = v4c.SIGNAL_TYPE_SCALAR
                if sig_dtype.kind in "SV":
                    sig_type = v4c.SIGNAL_TYPE_STRING
            else:
                if names in (v4c.CANOPEN_TIME_FIELDS, v4c.CANOPEN_DATE_FIELDS):
                    sig_type = v4c.SIGNAL_TYPE_CANOPEN
                elif names[0] != sig.name:
                    sig_type = v4c.SIGNAL_TYPE_STRUCTURE_COMPOSITION
                else:
                    sig_type = v4c.SIGNAL_TYPE_ARRAY

            gp_sig_types.append(sig_type)

            # first add the signals in the simple signal list
            if sig_type == v4c.SIGNAL_TYPE_SCALAR:

                # compute additional byte offset for large records size
                s_type, s_size = fmt_to_datatype_v4(sig_dtype, sig_shape)

                byte_size = s_size // 8 or 1

                if sig_dtype.kind == "u" and signal.bit_count <= 4:
                    s_size = signal.bit_count

                if signal.stream_sync:
                    channel_type = v4c.CHANNEL_TYPE_SYNC
                    if signal.attachment:
                        at_data, at_name, hash_sum = signal.attachment
                        attachment_addr = self.attach(
                            at_data, at_name, hash_sum, mime="video/avi", embedded=False
                        )
                        data_block_addr = attachment_addr
                    else:
                        data_block_addr = 0

                    sync_type = v4c.SYNC_TYPE_TIME
                else:
                    channel_type = v4c.CHANNEL_TYPE_VALUE
                    data_block_addr = 0
                    sync_type = v4c.SYNC_TYPE_NONE

                kwargs = {
                    "channel_type": channel_type,
                    "sync_type": sync_type,
                    "bit_count": s_size,
                    "byte_offset": offset,
                    "bit_offset": 0,
                    "data_type": s_type,
                    "data_block_addr": data_block_addr,
                    "flags": 0,
                }

                if signal.invalidation_bits is not None:
                    invalidation_bits = signal.invalidation_bits
                    kwargs["flags"] = v4c.FLAG_CN_INVALIDATION_PRESENT
                    kwargs["pos_invalidation_bit"] = 0
                else:
                    invalidation_bits = None

                ch = Channel(**kwargs)
                ch.name = name
                ch.unit = signal.unit
                ch.comment = signal.comment
                ch.display_name = signal.display_name

                # conversions for channel
                if signal.raw:
                    ch.conversion = conversion_transfer(signal.conversion, version=4)

                # source for channel
                source = signal.source
                if source:
                    if source in si_map:
                        ch.source = si_map[source]
                    else:
                        new_source = SourceInformation(
                            source_type=source.source_type, bus_type=source.bus_type
                        )
                        new_source.name = source.name
                        new_source.path = source.path
                        new_source.comment = source.comment

                        si_map[source] = new_source

                        ch.source = new_source

                gp_channels.append(ch)

                offset = byte_size

                gp_sdata.append(None)
                entry = (dg_cntr, ch_cntr)
                self.channels_db.add(name, entry)
                if ch.display_name:
                    self.channels_db.add(ch.display_name, entry)

                # update the parents as well
                parents[ch_cntr] = name, 0

                _shape = sig_shape[1:]
                types.append((name, sig_dtype, _shape))
                gp.single_channel_dtype = ch.dtype_fmt = dtype((sig_dtype, _shape))

                # simple channels don't have channel dependencies
                gp_dep.append(None)

            elif sig_type == v4c.SIGNAL_TYPE_CANOPEN:

                if names == v4c.CANOPEN_TIME_FIELDS:

                    types.append((name, "V6"))
                    gp.single_channel_dtype = dtype("V6")
                    byte_size = 6
                    s_type = v4c.DATA_TYPE_CANOPEN_TIME

                else:
                    vals = []
                    for field in ("ms", "min", "hour", "day", "month", "year"):
                        if field == "hour":
                            vals.append(
                                signal.samples[field]
                                + (signal.samples["summer_time"] << 7)
                            )
                        elif field == "day":
                            vals.append(
                                signal.samples[field]
                                + (signal.samples["day_of_week"] << 4)
                            )
                        else:
                            vals.append(signal.samples[field])
                    samples = fromarrays(vals)

                    types.append((name, "V7"))
                    gp.single_channel_dtype = dtype("V7")
                    byte_size = 7
                    s_type = v4c.DATA_TYPE_CANOPEN_DATE

                s_size = byte_size * 8

                # there is no channel dependency
                gp_dep.append(None)

                # add channel block
                kwargs = {
                    "channel_type": v4c.CHANNEL_TYPE_VALUE,
                    "bit_count": s_size,
                    "byte_offset": offset,
                    "bit_offset": 0,
                    "data_type": s_type,
                    "flags": 0,
                }
                if signal.invalidation_bits is not None:
                    invalidation_bits = signal.invalidation_bits
                    kwargs["flags"] = v4c.FLAG_CN_INVALIDATION_PRESENT
                    kwargs["pos_invalidation_bit"] = 0
                else:
                    invalidation_bits = None

                ch = Channel(**kwargs)
                ch.name = name
                ch.unit = signal.unit
                ch.comment = signal.comment
                ch.display_name = signal.display_name
                ch.dtype_fmt = gp.single_channel_dtype

                # source for channel
                source = signal.source
                if source:
                    if source in si_map:
                        ch.source = si_map[source]
                    else:
                        new_source = SourceInformation(
                            source_type=source.source_type, bus_type=source.bus_type
                        )
                        new_source.name = source.name
                        new_source.path = source.path
                        new_source.comment = source.comment

                        si_map[source] = new_source

                        ch.source = new_source

                gp_channels.append(ch)

                offset = byte_size

                entry = (dg_cntr, ch_cntr)
                self.channels_db.add(name, entry)
                if ch.display_name:
                    self.channels_db.add(ch.display_name, entry)

                # update the parents as well
                parents[ch_cntr] = name, 0

                gp_sdata.append(0)

            elif sig_type == v4c.SIGNAL_TYPE_STRUCTURE_COMPOSITION:
                (
                    offset,
                    dg_cntr,
                    ch_cntr,
                    struct_self,
                    new_fields,
                    new_types,
                ) = self._append_structure_composition_column_oriented(
                    gp,
                    signal,
                    field_names,
                    offset,
                    dg_cntr,
                    ch_cntr,
                    parents,
                    defined_texts,
                )

                if signal.invalidation_bits is not None:
                    invalidation_bits = signal.invalidation_bits
                else:
                    invalidation_bits = None

                gp["types"] = dtype(new_types)
                offset = gp["types"].itemsize

                samples = signal.samples

            elif sig_type == v4c.SIGNAL_TYPE_ARRAY:
                fields = []
                # here we have channel arrays or mdf v3 channel dependencies
                samples = signal.samples[names[0]]
                shape = samples.shape[1:]

                if len(names) > 1 or len(shape) > 1:
                    # add channel dependency block for composed parent channel
                    dims_nr = len(shape)
                    names_nr = len(names)

                    if names_nr == 0:
                        kwargs = {
                            "dims": dims_nr,
                            "ca_type": v4c.CA_TYPE_LOOKUP,
                            "flags": v4c.FLAG_CA_FIXED_AXIS,
                            "byte_offset_base": samples.dtype.itemsize,
                        }
                        for i in range(dims_nr):
                            kwargs[f"dim_size_{i}"] = shape[i]

                    elif len(names) == 1:
                        kwargs = {
                            "dims": dims_nr,
                            "ca_type": v4c.CA_TYPE_ARRAY,
                            "flags": 0,
                            "byte_offset_base": samples.dtype.itemsize,
                        }
                        for i in range(dims_nr):
                            kwargs[f"dim_size_{i}"] = shape[i]

                    else:
                        kwargs = {
                            "dims": dims_nr,
                            "ca_type": v4c.CA_TYPE_LOOKUP,
                            "flags": v4c.FLAG_CA_AXIS,
                            "byte_offset_base": samples.dtype.itemsize,
                        }
                        for i in range(dims_nr):
                            kwargs[f"dim_size_{i}"] = shape[i]

                    parent_dep = ChannelArrayBlock(**kwargs)
                    gp_dep.append([parent_dep])

                else:
                    # add channel dependency block for composed parent channel
                    kwargs = {
                        "dims": 1,
                        "ca_type": v4c.CA_TYPE_SCALE_AXIS,
                        "flags": 0,
                        "byte_offset_base": samples.dtype.itemsize,
                        "dim_size_0": shape[0],
                    }
                    parent_dep = ChannelArrayBlock(**kwargs)
                    gp_dep.append([parent_dep])

                field_name = field_names.get_unique_name(name)

                fields.append(samples)
                dtype_pair = field_name, samples.dtype, shape
                types.append(dtype_pair)

                # first we add the structure channel
                s_type, s_size = fmt_to_datatype_v4(samples.dtype, samples.shape, True)

                # add channel block
                kwargs = {
                    "channel_type": v4c.CHANNEL_TYPE_VALUE,
                    "bit_count": s_size,
                    "byte_offset": offset,
                    "bit_offset": 0,
                    "data_type": s_type,
                    "flags": 0,
                }

                if signal.invalidation_bits is not None:
                    invalidation_bits = signal.invalidation_bits
                    kwargs["flags"] = v4c.FLAG_CN_INVALIDATION_PRESENT
                    kwargs["pos_invalidation_bit"] = 0
                else:
                    invalidation_bits = None

                ch = Channel(**kwargs)
                ch.name = name
                ch.unit = signal.unit
                ch.comment = signal.comment
                ch.display_name = signal.display_name
                ch.dtype_fmt = samples.dtype

                # source for channel
                source = signal.source
                if source:
                    if source in si_map:
                        ch.source = si_map[source]
                    else:
                        new_source = SourceInformation(
                            source_type=source.source_type, bus_type=source.bus_type
                        )
                        new_source.name = source.name
                        new_source.path = source.path
                        new_source.comment = source.comment

                        si_map[source] = new_source

                        ch.source = new_source

                gp_channels.append(ch)

                size = s_size // 8
                for dim in shape:
                    size *= dim
                offset += size

                gp_sdata.append(None)
                entry = (dg_cntr, ch_cntr)
                self.channels_db.add(name, entry)
                if ch.display_name:
                    self.channels_db.add(ch.display_name, entry)

                # update the parents as well
                parents[ch_cntr] = name, 0

                ch_cntr += 1

                for name in names[1:]:
                    field_name = field_names.get_unique_name(name)

                    samples = signal.samples[name]
                    shape = samples.shape[1:]
                    fields.append(samples)
                    types.append((field_name, samples.dtype, shape))

                    # add channel dependency block
                    kwargs = {
                        "dims": 1,
                        "ca_type": v4c.CA_TYPE_SCALE_AXIS,
                        "flags": 0,
                        "byte_offset_base": samples.dtype.itemsize,
                        "dim_size_0": shape[0],
                    }
                    dep = ChannelArrayBlock(**kwargs)
                    gp_dep.append([dep])

                    # add components channel
                    s_type, s_size = fmt_to_datatype_v4(samples.dtype, ())
                    byte_size = s_size // 8 or 1
                    kwargs = {
                        "channel_type": v4c.CHANNEL_TYPE_VALUE,
                        "bit_count": s_size,
                        "byte_offset": offset,
                        "bit_offset": 0,
                        "data_type": s_type,
                        "flags": 0,
                    }

                    if signal.invalidation_bits is not None:
                        invalidation_bits = signal.invalidation_bits
                        kwargs["flags"] = v4c.FLAG_CN_INVALIDATION_PRESENT
                        kwargs["pos_invalidation_bit"] = 0
                    else:
                        invalidation_bits = None

                    ch = Channel(**kwargs)
                    ch.name = name
                    ch.unit = signal.unit
                    ch.comment = signal.comment
                    ch.display_name = signal.display_name
                    ch.dtype_fmt = samples.dtype

                    gp_channels.append(ch)

                    entry = dg_cntr, ch_cntr
                    parent_dep.axis_channels.append(entry)
                    for dim in shape:
                        byte_size *= dim
                    offset += byte_size

                    gp_sdata.append(None)
                    self.channels_db.add(name, entry)

                    # update the parents as well
                    parents[ch_cntr] = field_name, 0

                    ch_cntr += 1

                gp["types"] = dtype(types)

                samples = signal.samples

            else:

                encoding = signal.encoding
                samples = signal.samples
                sig_dtype = samples.dtype

                if encoding == "utf-8":
                    data_type = v4c.DATA_TYPE_STRING_UTF_8
                elif encoding == "latin-1":
                    data_type = v4c.DATA_TYPE_STRING_LATIN_1
                elif encoding == "utf-16-be":
                    data_type = v4c.DATA_TYPE_STRING_UTF_16_BE
                elif encoding == "utf-16-le":
                    data_type = v4c.DATA_TYPE_STRING_UTF_16_LE
                else:
                    raise MdfException(f'wrong encoding "{encoding}" for string signal')

                offsets = arange(len(samples), dtype=uint64) * (
                    signal.samples.itemsize + 4
                )

                values = [full(len(samples), samples.itemsize, dtype=uint32), samples]

                types_ = [("o", uint32), ("s", sig_dtype)]

                data = fromarrays(values, dtype=types_)

                data_size = len(data) * data.itemsize
                if data_size:
                    data_addr = tell()
                    info = SignalDataBlockInfo(
                        address=data_addr,
                        compressed_size=data_size,
                        original_size=data_size,
                        location=v4c.LOCATION_TEMPORARY_FILE,
                    )
                    gp_sdata.append([info])
                    data.tofile(file)
                else:
                    data_addr = 0
                    gp_sdata.append([])

                # compute additional byte offset for large records size
                byte_size = 8
                kwargs = {
                    "channel_type": v4c.CHANNEL_TYPE_VLSD,
                    "bit_count": 64,
                    "byte_offset": offset,
                    "bit_offset": 0,
                    "data_type": data_type,
                    "data_block_addr": data_addr,
                    "flags": 0,
                }

                if signal.invalidation_bits is not None:
                    invalidation_bits = signal.invalidation_bits
                    kwargs["flags"] = v4c.FLAG_CN_INVALIDATION_PRESENT
                    kwargs["pos_invalidation_bit"] = 0
                else:
                    invalidation_bits = None

                ch = Channel(**kwargs)
                ch.name = name
                ch.unit = signal.unit
                ch.comment = signal.comment
                ch.display_name = signal.display_name

                # conversions for channel
                conversion = conversion_transfer(signal.conversion, version=4)
                if signal.raw:
                    ch.conversion = conversion

                # source for channel
                source = signal.source
                if source:
                    if source in si_map:
                        ch.source = si_map[source]
                    else:
                        new_source = SourceInformation(
                            source_type=source.source_type, bus_type=source.bus_type
                        )
                        new_source.name = source.name
                        new_source.path = source.path
                        new_source.comment = source.comment

                        si_map[source] = new_source

                        ch.source = new_source

                gp_channels.append(ch)

                offset = byte_size

                entry = (dg_cntr, ch_cntr)
                self.channels_db.add(name, entry)
                if ch.display_name:
                    self.channels_db.add(ch.display_name, entry)

                # update the parents as well
                parents[ch_cntr] = name, 0

                types.append((name, uint64))
                gp.single_channel_dtype = ch.dtype_fmt = uint64

                samples = offsets

                # simple channels don't have channel dependencies
                gp_dep.append(None)

            gp.channel_group.samples_byte_nr = offset
            if invalidation_bits is not None:
                gp.channel_group.invalidation_bytes_nr = 1

            virtual_group.groups.append(dg_cntr)
            self.virtual_groups_map[dg_cntr] = cg_master_index

            virtual_group.record_size += offset
            if signal.invalidation_bits:
                virtual_group.record_size += 1

            dg_cntr += 1
            size = cycles_nr * samples.itemsize
            if size:

                data_address = tell()

                data = samples.tobytes()
                raw_size = len(data)
                data = lz_compress(data)

                size = len(data)
                write(data)

                gp.data_blocks.append(
                    DataBlockInfo(
                        address=data_address,
                        block_type=v4c.DZ_BLOCK_LZ,
                        original_size=raw_size,
                        compressed_size=size,
                        param=0,
                    )
                )

                if invalidation_bits is not None:
                    addr = tell()
                    data = invalidation_bits.tobytes()
                    raw_size = len(data)
                    data = lz_compress(data)
                    size = len(data)
                    write(data)

                    gp.data_blocks[-1].invalidation_block(
                        InvalidationBlockInfo(
                            address=addr,
                            block_type=v4c.DZ_BLOCK_LZ,
                            original_size=raw_size,
                            compressed_size=size,
                            param=None,
                        )
                    )

            gp.data_location = v4c.LOCATION_TEMPORARY_FILE

        return initial_dg_cntr

    def _append_dataframe(
        self, df, acq_name=None, acq_source=None, comment=None, units=None
    ):
        """
        Appends a new data group from a Pandas data frame.

        """
        units = units or {}

        if df.shape == (0, 0):
            return

        t = df.index
        index_name = df.index.name
        time_name = index_name or "time"
        sync_type = v4c.SYNC_TYPE_TIME
        time_unit = "s"

        dg_cntr = len(self.groups)

        gp = Group(None)
        gp.signal_data = gp_sdata = []
        gp.channels = gp_channels = []
        gp.channel_dependencies = gp_dep = []
        gp.signal_types = gp_sig_types = []

        cycles_nr = len(t)

        # channel group
        kwargs = {"cycles_nr": cycles_nr, "samples_byte_nr": 0}
        gp.channel_group = ChannelGroup(**kwargs)
        gp.channel_group.acq_name = acq_name
        gp.channel_group.acq_source = acq_source
        gp.channel_group.comment = comment

        self.groups.append(gp)

        fields = []
        types = []
        parents = {}
        ch_cntr = 0
        offset = 0
        field_names = UniqueDB()

        # setup all blocks related to the time master channel

        file = self._tempfile
        tell = file.tell
        seek = file.seek

        seek(0, 2)

        virtual_group = VirtualChannelGroup()
        self.virtual_groups[dg_cntr] = virtual_group
        self.virtual_groups_map[dg_cntr] = dg_cntr
        virtual_group.groups.append(dg_cntr)
        virtual_group.cycles_nr = cycles_nr

        # time channel
        t_type, t_size = fmt_to_datatype_v4(t.dtype, t.shape)
        kwargs = {
            "channel_type": v4c.CHANNEL_TYPE_MASTER,
            "data_type": t_type,
            "sync_type": sync_type,
            "byte_offset": 0,
            "bit_offset": 0,
            "bit_count": t_size,
            "min_raw_value": t[0] if cycles_nr else 0,
            "max_raw_value": t[-1] if cycles_nr else 0,
            "lower_limit": t[0] if cycles_nr else 0,
            "upper_limit": t[-1] if cycles_nr else 0,
            "flags": v4c.FLAG_PHY_RANGE_OK | v4c.FLAG_VAL_RANGE_OK,
        }
        ch = Channel(**kwargs)
        ch.unit = time_unit
        ch.name = time_name
        ch.dtype_fmt = t.dtype
        name = time_name
        gp_channels.append(ch)

        gp_sdata.append(None)
        self.channels_db.add(name, (dg_cntr, ch_cntr))
        self.masters_db[dg_cntr] = 0
        # data group record parents
        parents[ch_cntr] = name, 0

        # time channel doesn't have channel dependencies
        gp_dep.append(None)

        fields.append(t)
        types.append((name, t.dtype))
        field_names.get_unique_name(name)

        offset += t_size // 8
        ch_cntr += 1

        gp_sig_types.append(0)

        for signal in df:
            if index_name == signal:
                continue

            sig = df[signal]
            name = signal

            sig_type = v4c.SIGNAL_TYPE_SCALAR
            if sig.dtype.kind in "SV":
                sig_type = v4c.SIGNAL_TYPE_STRING

            gp_sig_types.append(sig_type)

            # first add the signals in the simple signal list
            if sig_type == v4c.SIGNAL_TYPE_SCALAR:

                # compute additional byte offset for large records size
                if sig.dtype.kind == "O":
                    sig = encode(sig.values.astype(str), "utf-8")

                s_type, s_size = fmt_to_datatype_v4(sig.dtype, sig.shape)

                byte_size = s_size // 8 or 1

                channel_type = v4c.CHANNEL_TYPE_VALUE
                data_block_addr = 0
                sync_type = v4c.SYNC_TYPE_NONE

                kwargs = {
                    "channel_type": channel_type,
                    "sync_type": sync_type,
                    "bit_count": s_size,
                    "byte_offset": offset,
                    "bit_offset": 0,
                    "data_type": s_type,
                    "data_block_addr": data_block_addr,
                }

                ch = Channel(**kwargs)
                ch.name = name
                ch.unit = units.get(name, "")
                ch.dtype_fmt = dtype((sig.dtype, sig.shape[1:]))

                gp_channels.append(ch)

                offset += byte_size

                gp_sdata.append(None)
                self.channels_db.add(name, (dg_cntr, ch_cntr))

                # update the parents as well
                field_name = field_names.get_unique_name(name)
                parents[ch_cntr] = field_name, 0

                fields.append(sig)
                types.append((field_name, sig.dtype, sig.shape[1:]))

                ch_cntr += 1

                # simple channels don't have channel dependencies
                gp_dep.append(None)

            elif sig_type == v4c.SIGNAL_TYPE_STRING:
                offsets = arange(len(sig), dtype=uint64) * (sig.dtype.itemsize + 4)

                values = [full(len(sig), sig.dtype.itemsize, dtype=uint32), sig.values]

                types_ = [("", uint32), ("", sig.dtype)]

                data = fromarrays(values, dtype=types_)

                data_size = len(data) * data.itemsize
                if data_size:
                    data_addr = tell()
                    info = SignalDataBlockInfo(
                        address=data_addr,
                        compressed_size=data_size,
                        original_size=data_size,
                        location=v4c.LOCATION_TEMPORARY_FILE,
                    )
                    gp_sdata.append([info])
                    data.tofile(file)
                else:
                    data_addr = 0
                    gp_sdata.append([])

                # compute additional byte offset for large records size
                byte_size = 8
                kwargs = {
                    "channel_type": v4c.CHANNEL_TYPE_VLSD,
                    "bit_count": 64,
                    "byte_offset": offset,
                    "bit_offset": 0,
                    "data_type": v4c.DATA_TYPE_STRING_UTF_8,
                    "min_raw_value": 0,
                    "max_raw_value": 0,
                    "lower_limit": 0,
                    "upper_limit": 0,
                    "flags": 0,
                    "data_block_addr": data_addr,
                }

                ch = Channel(**kwargs)
                ch.name = name
                ch.unit = units.get(name, "")
                ch.dtype_fmt = dtype("<u8")

                gp_channels.append(ch)

                offset += byte_size

                self.channels_db.add(name, (dg_cntr, ch_cntr))

                # update the parents as well
                field_name = field_names.get_unique_name(name)
                parents[ch_cntr] = field_name, 0

                fields.append(offsets)
                types.append((field_name, uint64))

                ch_cntr += 1

                # simple channels don't have channel dependencies
                gp_dep.append(None)

        virtual_group.record_size = offset
        virtual_group.cycles_nr = cycles_nr

        gp.channel_group.cycles_nr = cycles_nr
        gp.channel_group.samples_byte_nr = offset

        # data group
        gp.data_group = DataGroup()

        # data block
        types = dtype(types)

        gp.sorted = True
        gp.types = types
        gp.parents = parents

        if df.shape[0]:
            samples = fromarrays(fields, dtype=types)
        else:
            samples = array([])

        size = len(samples) * samples.itemsize
        if size:
            data_address = self._tempfile.tell()
            gp.data_location = v4c.LOCATION_TEMPORARY_FILE

            samples.tofile(self._tempfile)

            self._tempfile.write(samples.tobytes())

            gp.data_blocks.append(
                DataBlockInfo(
                    address=data_address,
                    block_type=v4c.DT_BLOCK,
                    original_size=size,
                    compressed_size=size,
                    param=0,
                )
            )
        else:
            gp.data_location = v4c.LOCATION_TEMPORARY_FILE

    def _append_structure_composition(
        self,
        grp,
        signal,
        field_names,
        offset,
        dg_cntr,
        ch_cntr,
        parents,
        defined_texts,
        invalidation_bytes_nr,
        inval_bits,
        inval_cntr,
    ):
        si_map = self._si_map

        fields = []
        types = []

        file = self._tempfile
        seek = file.seek
        seek(0, 2)

        gp = grp
        gp_sdata = gp.signal_data
        gp_channels = gp.channels
        gp_dep = gp.channel_dependencies

        name = signal.name
        names = signal.samples.dtype.names

        field_name = field_names.get_unique_name(name)

        # first we add the structure channel

        if signal.attachment and signal.attachment[0]:
            at_data, at_name, hash_sum = signal.attachment
            if at_name is not None:
                suffix = Path(at_name).suffix.lower().strip(".")
            else:
                suffix = "dbc"
            if suffix == "a2l":
                mime = "applciation/A2L"
            else:
                mime = f"application/x-{suffix}"
            attachment_index = self.attach(
                at_data, at_name, hash_sum=hash_sum, mime=mime
            )
            attachment = attachment_index
        else:
            attachment = None

        # add channel block
        kwargs = {
            "channel_type": v4c.CHANNEL_TYPE_VALUE,
            "bit_count": signal.samples.dtype.itemsize * 8,
            "byte_offset": offset,
            "bit_offset": 0,
            "data_type": v4c.DATA_TYPE_BYTEARRAY,
            "precision": 0,
        }

        if attachment is not None:
            kwargs["attachment_addr"] = 0

        source_bus = signal.source and signal.source.source_type == v4c.SOURCE_BUS

        if source_bus:
            kwargs["flags"] = v4c.FLAG_CN_BUS_EVENT
            flags_ = v4c.FLAG_CN_BUS_EVENT
            grp.channel_group.flags |= (
                v4c.FLAG_CG_BUS_EVENT | v4c.FLAG_CG_PLAIN_BUS_EVENT
            )
        else:
            kwargs["flags"] = 0
            flags_ = 0

        if invalidation_bytes_nr and signal.invalidation_bits is not None:
            inval_bits.append(signal.invalidation_bits)
            kwargs["flags"] |= v4c.FLAG_CN_INVALIDATION_PRESENT
            kwargs["pos_invalidation_bit"] = inval_cntr
            inval_cntr += 1

        ch = Channel(**kwargs)
        ch.name = name
        ch.unit = signal.unit
        ch.comment = signal.comment
        ch.display_name = signal.display_name
        ch.attachment = attachment
        ch.dtype_fmt = signal.samples.dtype

        if source_bus and grp.channel_group.acq_source is None:
            grp.channel_group.acq_source = SourceInformation.from_common_source(
                signal.source
            )

            if signal.source.bus_type == v4c.BUS_TYPE_CAN:
                grp.channel_group.path_separator = 46
                grp.channel_group.acq_name = "CAN"
            elif signal.source.bus_type == v4c.BUS_TYPE_FLEXRAY:
                grp.channel_group.path_separator = 46
                grp.channel_group.acq_name = "FLEXRAY"
            elif signal.source.bus_type == v4c.BUS_TYPE_ETHERNET:
                grp.channel_group.path_separator = 46
                grp.channel_group.acq_name = "ETHERNET"
            elif signal.source.bus_type == v4c.BUS_TYPE_K_LINE:
                grp.channel_group.path_separator = 46
                grp.channel_group.acq_name = "K_LINE"
            elif signal.source.bus_type == v4c.BUS_TYPE_MOST:
                grp.channel_group.path_separator = 46
                grp.channel_group.acq_name = "MOST"
            elif signal.source.bus_type == v4c.BUS_TYPE_LIN:
                grp.channel_group.path_separator = 46
                grp.channel_group.acq_name = "LIN"

        # source for channel
        source = signal.source
        if source:
            if source in si_map:
                ch.source = si_map[source]
            else:
                new_source = SourceInformation(
                    source_type=source.source_type, bus_type=source.bus_type
                )
                new_source.name = source.name
                new_source.path = source.path
                new_source.comment = source.comment

                si_map[source] = new_source

                ch.source = new_source

        entry = dg_cntr, ch_cntr
        gp_channels.append(ch)
        struct_self = entry

        gp_sdata.append(None)
        self.channels_db.add(name, entry)
        if ch.display_name:
            self.channels_db.add(ch.display_name, entry)

        # update the parents as well
        parents[ch_cntr] = name, 0

        ch_cntr += 1

        dep_list = []
        gp_dep.append(dep_list)

        # then we add the fields
        for name in names:
            field_name = field_names.get_unique_name(name)

            samples = signal.samples[name]
            fld_names = samples.dtype.names

            if fld_names is None:
                sig_type = v4c.SIGNAL_TYPE_SCALAR
                if samples.dtype.kind in "SV":
                    sig_type = v4c.SIGNAL_TYPE_STRING
            else:
                if fld_names in (v4c.CANOPEN_TIME_FIELDS, v4c.CANOPEN_DATE_FIELDS):
                    sig_type = v4c.SIGNAL_TYPE_CANOPEN
                elif fld_names[0] != name:
                    sig_type = v4c.SIGNAL_TYPE_STRUCTURE_COMPOSITION
                else:
                    sig_type = v4c.SIGNAL_TYPE_ARRAY

            if sig_type in (v4c.SIGNAL_TYPE_SCALAR, v4c.SIGNAL_TYPE_STRING):

                s_type, s_size = fmt_to_datatype_v4(samples.dtype, samples.shape)
                byte_size = s_size // 8 or 1

                fields.append(samples)
                types.append((field_name, samples.dtype, samples.shape[1:]))

                # add channel block
                kwargs = {
                    "channel_type": v4c.CHANNEL_TYPE_VALUE,
                    "bit_count": s_size,
                    "byte_offset": offset,
                    "bit_offset": 0,
                    "data_type": s_type,
                    "flags": flags_,
                }

                if invalidation_bytes_nr:
                    if signal.invalidation_bits is not None:
                        inval_bits.append(signal.invalidation_bits)
                        kwargs["flags"] |= v4c.FLAG_CN_INVALIDATION_PRESENT
                        kwargs["pos_invalidation_bit"] = inval_cntr
                        inval_cntr += 1

                ch = Channel(**kwargs)
                ch.name = name
                ch.dtype_fmt = dtype((samples.dtype, samples.shape[1:]))

                entry = (dg_cntr, ch_cntr)
                gp_channels.append(ch)
                dep_list.append(entry)

                offset += byte_size

                gp_sdata.append(None)
                self.channels_db.add(name, entry)

                # update the parents as well
                parents[ch_cntr] = field_name, 0

                ch_cntr += 1
                gp_dep.append(None)

            elif sig_type == v4c.SIGNAL_TYPE_ARRAY:
                # here we have channel arrays or mdf v3 channel dependencies
                array_samples = samples
                names = samples.dtype.names
                samples = array_samples[names[0]]
                shape = samples.shape[1:]

                if len(names) > 1:
                    # add channel dependency block for composed parent channel
                    dims_nr = len(shape)
                    names_nr = len(names)

                    if names_nr == 0:
                        kwargs = {
                            "dims": dims_nr,
                            "ca_type": v4c.CA_TYPE_LOOKUP,
                            "flags": v4c.FLAG_CA_FIXED_AXIS,
                            "byte_offset_base": samples.dtype.itemsize,
                        }
                        for i in range(dims_nr):
                            kwargs[f"dim_size_{i}"] = shape[i]

                    elif len(names) == 1:
                        kwargs = {
                            "dims": dims_nr,
                            "ca_type": v4c.CA_TYPE_ARRAY,
                            "flags": 0,
                            "byte_offset_base": samples.dtype.itemsize,
                        }
                        for i in range(dims_nr):
                            kwargs[f"dim_size_{i}"] = shape[i]

                    else:
                        kwargs = {
                            "dims": dims_nr,
                            "ca_type": v4c.CA_TYPE_LOOKUP,
                            "flags": v4c.FLAG_CA_AXIS,
                            "byte_offset_base": samples.dtype.itemsize,
                        }
                        for i in range(dims_nr):
                            kwargs[f"dim_size_{i}"] = shape[i]

                    parent_dep = ChannelArrayBlock(**kwargs)
                    gp_dep.append([parent_dep])

                else:
                    # add channel dependency block for composed parent channel
                    kwargs = {
                        "dims": 1,
                        "ca_type": v4c.CA_TYPE_SCALE_AXIS,
                        "flags": 0,
                        "byte_offset_base": samples.dtype.itemsize,
                        "dim_size_0": shape[0],
                    }
                    parent_dep = ChannelArrayBlock(**kwargs)
                    gp_dep.append([parent_dep])

                field_name = field_names.get_unique_name(name)

                fields.append(samples)
                dtype_pair = field_name, samples.dtype, shape
                types.append(dtype_pair)

                # first we add the structure channel
                s_type, s_size = fmt_to_datatype_v4(samples.dtype, samples.shape, True)

                # add channel block
                kwargs = {
                    "channel_type": v4c.CHANNEL_TYPE_VALUE,
                    "bit_count": s_size,
                    "byte_offset": offset,
                    "bit_offset": 0,
                    "data_type": s_type,
                    "flags": 0,
                }

                if invalidation_bytes_nr:
                    if signal.invalidation_bits is not None:
                        inval_bits.append(signal.invalidation_bits)
                        kwargs["flags"] |= v4c.FLAG_CN_INVALIDATION_PRESENT
                        kwargs["pos_invalidation_bit"] = inval_cntr
                        inval_cntr += 1

                ch = Channel(**kwargs)
                ch.name = name
                ch.unit = signal.unit
                ch.comment = signal.comment
                ch.display_name = signal.display_name
                ch.dtype_fmt = samples.dtype

                # source for channel
                source = signal.source
                if source:
                    if source in si_map:
                        ch.source = si_map[source]
                    else:
                        new_source = SourceInformation(
                            source_type=source.source_type, bus_type=source.bus_type
                        )
                        new_source.name = source.name
                        new_source.path = source.path
                        new_source.comment = source.comment

                        si_map[source] = new_source

                        ch.source = new_source

                gp_channels.append(ch)

                size = s_size // 8
                for dim in shape:
                    size *= dim
                offset += size

                gp_sdata.append(None)
                entry = (dg_cntr, ch_cntr)
                self.channels_db.add(name, entry)
                if ch.display_name:
                    self.channels_db.add(ch.display_name, entry)

                # update the parents as well
                parents[ch_cntr] = name, 0

                ch_cntr += 1

                for name in names[1:]:
                    field_name = field_names.get_unique_name(name)

                    samples = array_samples[name]
                    shape = samples.shape[1:]
                    fields.append(samples)
                    types.append((field_name, samples.dtype, shape))

                    # add channel dependency block
                    kwargs = {
                        "dims": 1,
                        "ca_type": v4c.CA_TYPE_SCALE_AXIS,
                        "flags": 0,
                        "byte_offset_base": samples.dtype.itemsize,
                        "dim_size_0": shape[0],
                    }
                    dep = ChannelArrayBlock(**kwargs)
                    gp_dep.append([dep])

                    # add components channel
                    s_type, s_size = fmt_to_datatype_v4(samples.dtype, ())
                    byte_size = s_size // 8 or 1
                    kwargs = {
                        "channel_type": v4c.CHANNEL_TYPE_VALUE,
                        "bit_count": s_size,
                        "byte_offset": offset,
                        "bit_offset": 0,
                        "data_type": s_type,
                        "flags": 0,
                    }

                    if invalidation_bytes_nr:
                        if signal.invalidation_bits is not None:
                            inval_bits.append(signal.invalidation_bits)
                            kwargs["flags"] |= v4c.FLAG_CN_INVALIDATION_PRESENT
                            kwargs["pos_invalidation_bit"] = inval_cntr
                            inval_cntr += 1

                    ch = Channel(**kwargs)
                    ch.name = name
                    ch.unit = signal.unit
                    ch.comment = signal.comment
                    ch.display_name = signal.display_name
                    ch.dtype_fmt = samples.dtype

                    gp_channels.append(ch)

                    entry = dg_cntr, ch_cntr
                    parent_dep.axis_channels.append(entry)
                    for dim in shape:
                        byte_size *= dim
                    offset += byte_size

                    gp_sdata.append(None)
                    self.channels_db.add(name, entry)

                    # update the parents as well
                    parents[ch_cntr] = field_name, 0

                    ch_cntr += 1

            elif sig_type == v4c.SIGNAL_TYPE_STRUCTURE_COMPOSITION:
                struct = Signal(
                    samples,
                    samples,
                    name=name,
                    invalidation_bits=signal.invalidation_bits,
                )
                (
                    offset,
                    dg_cntr,
                    ch_cntr,
                    sub_structure,
                    new_fields,
                    new_types,
                    inval_cntr,
                ) = self._append_structure_composition(
                    grp,
                    struct,
                    field_names,
                    offset,
                    dg_cntr,
                    ch_cntr,
                    parents,
                    defined_texts,
                    invalidation_bytes_nr,
                    inval_bits,
                    inval_cntr,
                )
                dep_list.append(sub_structure)
                fields.extend(new_fields)
                types.extend(new_types)

        return offset, dg_cntr, ch_cntr, struct_self, fields, types, inval_cntr

    def _append_structure_composition_column_oriented(
        self,
        grp,
        signal,
        field_names,
        offset,
        dg_cntr,
        ch_cntr,
        parents,
        defined_texts,
    ):
        si_map = self._si_map

        fields = []
        types = []

        file = self._tempfile
        seek = file.seek
        seek(0, 2)

        gp = grp
        gp_sdata = gp.signal_data
        gp_channels = gp.channels
        gp_dep = gp.channel_dependencies

        name = signal.name
        names = signal.samples.dtype.names

        field_name = field_names.get_unique_name(name)

        # first we add the structure channel

        if signal.attachment and signal.attachment[0]:
            at_data, at_name, hash_sum = signal.attachment
            if at_name is not None:
                suffix = Path(at_name).suffix.strip(".")
            else:
                suffix = "dbc"
            attachment_index = self.attach(
                at_data, at_name, hash_sum=hash_sum, mime=f"application/x-{suffix}"
            )
            attachment = attachment_index
        else:
            attachment = None

        # add channel block
        kwargs = {
            "channel_type": v4c.CHANNEL_TYPE_VALUE,
            "bit_count": signal.samples.dtype.itemsize * 8,
            "byte_offset": offset,
            "bit_offset": 0,
            "data_type": v4c.DATA_TYPE_BYTEARRAY,
            "precision": 0,
        }

        if attachment is not None:
            kwargs["attachment_addr"] = 0

        source_bus = signal.source and signal.source.source_type == v4c.SOURCE_BUS

        if source_bus:
            kwargs["flags"] = v4c.FLAG_CN_BUS_EVENT
            flags_ = v4c.FLAG_CN_BUS_EVENT
            grp.channel_group.flags |= v4c.FLAG_CG_BUS_EVENT
        else:
            kwargs["flags"] = 0
            flags_ = 0

        if signal.invalidation_bits is not None:
            kwargs["flags"] |= v4c.FLAG_CN_INVALIDATION_PRESENT
            kwargs["pos_invalidation_bit"] = 0

        ch = Channel(**kwargs)
        ch.name = name
        ch.unit = signal.unit
        ch.comment = signal.comment
        ch.display_name = signal.display_name
        ch.attachment = attachment
        ch.dtype_fmt = signal.samples.dtype

        if source_bus:
            grp.channel_group.acq_source = SourceInformation.from_common_source(
                signal.source
            )

            if signal.source.bus_type == v4c.BUS_TYPE_CAN:
                grp.channel_group.path_separator = 46
                grp.channel_group.acq_name = "CAN"
            elif signal.source.bus_type == v4c.BUS_TYPE_FLEXRAY:
                grp.channel_group.path_separator = 46
                grp.channel_group.acq_name = "FLEXRAY"
            elif signal.source.bus_type == v4c.BUS_TYPE_ETHERNET:
                grp.channel_group.path_separator = 46
                grp.channel_group.acq_name = "ETHERNET"

        # source for channel
        source = signal.source
        if source:
            if source in si_map:
                ch.source = si_map[source]
            else:
                new_source = SourceInformation(
                    source_type=source.source_type, bus_type=source.bus_type
                )
                new_source.name = source.name
                new_source.path = source.path
                new_source.comment = source.comment

                si_map[source] = new_source

                ch.source = new_source

        entry = dg_cntr, ch_cntr
        gp_channels.append(ch)
        struct_self = entry

        gp_sdata.append(None)
        self.channels_db.add(name, entry)
        if ch.display_name:
            self.channels_db.add(ch.display_name, entry)

        # update the parents as well
        parents[ch_cntr] = name, 0

        ch_cntr += 1

        dep_list = []
        gp_dep.append(dep_list)

        # then we add the fields

        for name in names:
            field_name = field_names.get_unique_name(name)

            samples = signal.samples[name]
            fld_names = samples.dtype.names

            if fld_names is None:
                sig_type = v4c.SIGNAL_TYPE_SCALAR
                if samples.dtype.kind in "SV":
                    sig_type = v4c.SIGNAL_TYPE_STRING
            else:
                if fld_names in (v4c.CANOPEN_TIME_FIELDS, v4c.CANOPEN_DATE_FIELDS):
                    sig_type = v4c.SIGNAL_TYPE_CANOPEN
                elif fld_names[0] != name:
                    sig_type = v4c.SIGNAL_TYPE_STRUCTURE_COMPOSITION
                else:
                    sig_type = v4c.SIGNAL_TYPE_ARRAY

            if sig_type in (v4c.SIGNAL_TYPE_SCALAR, v4c.SIGNAL_TYPE_STRING):

                s_type, s_size = fmt_to_datatype_v4(samples.dtype, samples.shape)
                byte_size = s_size // 8 or 1

                fields.append(samples)
                types.append((field_name, samples.dtype, samples.shape[1:]))

                # add channel block
                kwargs = {
                    "channel_type": v4c.CHANNEL_TYPE_VALUE,
                    "bit_count": s_size,
                    "byte_offset": offset,
                    "bit_offset": 0,
                    "data_type": s_type,
                    "flags": flags_,
                }

                if signal.invalidation_bits is not None:
                    kwargs["flags"] |= v4c.FLAG_CN_INVALIDATION_PRESENT
                    kwargs["pos_invalidation_bit"] = 0

                ch = Channel(**kwargs)
                ch.name = name
                ch.dtype_fmt = dtype((samples.dtype, samples.shape[1:]))

                entry = (dg_cntr, ch_cntr)
                gp_channels.append(ch)
                dep_list.append(entry)

                offset += byte_size

                gp_sdata.append(None)
                self.channels_db.add(name, entry)

                # update the parents as well
                parents[ch_cntr] = field_name, 0

                ch_cntr += 1
                gp_dep.append(None)

            elif sig_type == v4c.SIGNAL_TYPE_ARRAY:
                # here we have channel arrays or mdf v3 channel dependencies
                array_samples = samples
                names = samples.dtype.names
                samples = array_samples[names[0]]
                shape = samples.shape[1:]

                if len(names) > 1:
                    # add channel dependency block for composed parent channel
                    dims_nr = len(shape)
                    names_nr = len(names)

                    if names_nr == 0:
                        kwargs = {
                            "dims": dims_nr,
                            "ca_type": v4c.CA_TYPE_LOOKUP,
                            "flags": v4c.FLAG_CA_FIXED_AXIS,
                            "byte_offset_base": samples.dtype.itemsize,
                        }
                        for i in range(dims_nr):
                            kwargs[f"dim_size_{i}"] = shape[i]

                    elif len(names) == 1:
                        kwargs = {
                            "dims": dims_nr,
                            "ca_type": v4c.CA_TYPE_ARRAY,
                            "flags": 0,
                            "byte_offset_base": samples.dtype.itemsize,
                        }
                        for i in range(dims_nr):
                            kwargs[f"dim_size_{i}"] = shape[i]

                    else:
                        kwargs = {
                            "dims": dims_nr,
                            "ca_type": v4c.CA_TYPE_LOOKUP,
                            "flags": v4c.FLAG_CA_AXIS,
                            "byte_offset_base": samples.dtype.itemsize,
                        }
                        for i in range(dims_nr):
                            kwargs[f"dim_size_{i}"] = shape[i]

                    parent_dep = ChannelArrayBlock(**kwargs)
                    gp_dep.append([parent_dep])

                else:
                    # add channel dependency block for composed parent channel
                    kwargs = {
                        "dims": 1,
                        "ca_type": v4c.CA_TYPE_SCALE_AXIS,
                        "flags": 0,
                        "byte_offset_base": samples.dtype.itemsize,
                        "dim_size_0": shape[0],
                    }
                    parent_dep = ChannelArrayBlock(**kwargs)
                    gp_dep.append([parent_dep])

                field_name = field_names.get_unique_name(name)

                fields.append(samples)
                dtype_pair = field_name, samples.dtype, shape
                types.append(dtype_pair)

                # first we add the structure channel
                s_type, s_size = fmt_to_datatype_v4(samples.dtype, samples.shape, True)

                # add channel block
                kwargs = {
                    "channel_type": v4c.CHANNEL_TYPE_VALUE,
                    "bit_count": s_size,
                    "byte_offset": offset,
                    "bit_offset": 0,
                    "data_type": s_type,
                    "flags": 0,
                }

                if signal.invalidation_bits is not None:
                    kwargs["flags"] |= v4c.FLAG_CN_INVALIDATION_PRESENT
                    kwargs["pos_invalidation_bit"] = 0

                ch = Channel(**kwargs)
                ch.name = name
                ch.unit = signal.unit
                ch.comment = signal.comment
                ch.display_name = signal.display_name
                ch.dtype_fmt = samples.dtype

                # source for channel
                source = signal.source
                if source:
                    if source in si_map:
                        ch.source = si_map[source]
                    else:
                        new_source = SourceInformation(
                            source_type=source.source_type, bus_type=source.bus_type
                        )
                        new_source.name = source.name
                        new_source.path = source.path
                        new_source.comment = source.comment

                        si_map[source] = new_source

                        ch.source = new_source

                gp_channels.append(ch)

                size = s_size // 8
                for dim in shape:
                    size *= dim
                offset += size

                gp_sdata.append(None)
                entry = (dg_cntr, ch_cntr)
                self.channels_db.add(name, entry)
                if ch.display_name:
                    self.channels_db.add(ch.display_name, entry)

                # update the parents as well
                parents[ch_cntr] = name, 0

                ch_cntr += 1

                for name in names[1:]:
                    field_name = field_names.get_unique_name(name)

                    samples = array_samples[name]
                    shape = samples.shape[1:]
                    fields.append(samples)
                    types.append((field_name, samples.dtype, shape))

                    # add channel dependency block
                    kwargs = {
                        "dims": 1,
                        "ca_type": v4c.CA_TYPE_SCALE_AXIS,
                        "flags": 0,
                        "byte_offset_base": samples.dtype.itemsize,
                        "dim_size_0": shape[0],
                    }
                    dep = ChannelArrayBlock(**kwargs)
                    gp_dep.append([dep])

                    # add components channel
                    s_type, s_size = fmt_to_datatype_v4(samples.dtype, ())
                    byte_size = s_size // 8 or 1
                    kwargs = {
                        "channel_type": v4c.CHANNEL_TYPE_VALUE,
                        "bit_count": s_size,
                        "byte_offset": offset,
                        "bit_offset": 0,
                        "data_type": s_type,
                        "flags": 0,
                    }

                    if signal.invalidation_bits is not None:
                        kwargs["flags"] |= v4c.FLAG_CN_INVALIDATION_PRESENT
                        kwargs["pos_invalidation_bit"] = 0

                    ch = Channel(**kwargs)
                    ch.name = name
                    ch.unit = signal.unit
                    ch.comment = signal.comment
                    ch.display_name = signal.display_name
                    ch.dtype_fmt = samples.dtype

                    gp_channels.append(ch)

                    entry = dg_cntr, ch_cntr
                    parent_dep.axis_channels.append(entry)
                    for dim in shape:
                        byte_size *= dim
                    offset += byte_size

                    gp_sdata.append(None)
                    self.channels_db.add(name, entry)

                    # update the parents as well
                    parents[ch_cntr] = field_name, 0

                    ch_cntr += 1

            elif sig_type == v4c.SIGNAL_TYPE_STRUCTURE_COMPOSITION:
                struct = Signal(
                    samples,
                    samples,
                    name=name,
                    invalidation_bits=signal.invalidation_bits,
                )
                (
                    offset,
                    dg_cntr,
                    ch_cntr,
                    sub_structure,
                    new_fields,
                    new_types,
                ) = self._append_structure_composition_column_oriented(
                    grp,
                    struct,
                    field_names,
                    offset,
                    dg_cntr,
                    ch_cntr,
                    parents,
                    defined_texts,
                )
                dep_list.append(sub_structure)
                fields.extend(new_fields)
                types.extend(new_types)

        return offset, dg_cntr, ch_cntr, struct_self, fields, types

    def extend(self, index, signals):
        """
        Extend a group with new samples. *signals* contains (values, invalidation_bits)
        pairs for each extended signal. The first pair is the master channel's pair, and the
        next pairs must respect the same order in which the signals were appended. The samples must have raw
        or physical values according to the *Signals* used for the initial append.

        Parameters
        ----------
        index : int
            group index
        signals : list
            list on (numpy.ndarray, numpy.ndarray) objects

        Examples
        --------
        >>> # case 1 conversion type None
        >>> s1 = np.array([1, 2, 3, 4, 5])
        >>> s2 = np.array([-1, -2, -3, -4, -5])
        >>> s3 = np.array([0.1, 0.04, 0.09, 0.16, 0.25])
        >>> t = np.array([0.001, 0.002, 0.003, 0.004, 0.005])
        >>> names = ['Positive', 'Negative', 'Float']
        >>> units = ['+', '-', '.f']
        >>> s1 = Signal(samples=s1, timestamps=t, unit='+', name='Positive')
        >>> s2 = Signal(samples=s2, timestamps=t, unit='-', name='Negative')
        >>> s3 = Signal(samples=s3, timestamps=t, unit='flts', name='Floats')
        >>> mdf = MDF4('new.mdf')
        >>> mdf.append([s1, s2, s3], comment='created by asammdf v1.1.0')
        >>> t = np.array([0.006, 0.007, 0.008, 0.009, 0.010])
        >>> # extend without invalidation bits
        >>> mdf2.extend(0, [(t, None), (s1, None), (s2, None), (s3, None)])
        >>> # some invaldiation btis
        >>> s1_inv = np.array([0,0,0,1,1], dtype=np.bool)
        >>> mdf2.extend(0, [(t, None), (s1.samples, None), (s2.samples, None), (s3.samples, None)])

        """
        if self.version >= "4.20" and (self._column_storage or 1):
            return self._extend_column_oriented(index, signals)
        gp = self.groups[index]
        if not signals:
            message = '"append" requires a non-empty list of Signal objects'
            raise MdfException(message)

        stream = self._tempfile

        fields = []
        types = []
        inval_bits = []

        added_cycles = len(signals[0][0])

        invalidation_bytes_nr = gp.channel_group.invalidation_bytes_nr
        for i, ((signal, invalidation_bits), sig_type) in enumerate(
            zip(signals, gp.signal_types)
        ):

            # first add the signals in the simple signal list
            if sig_type == v4c.SIGNAL_TYPE_SCALAR:

                fields.append(signal)
                types.append(("", signal.dtype, signal.shape[1:]))

                if invalidation_bytes_nr and invalidation_bits is not None:
                    inval_bits.append(invalidation_bits)

            elif sig_type == v4c.SIGNAL_TYPE_CANOPEN:
                names = signal.dtype.names

                if names == v4c.CANOPEN_TIME_FIELDS:

                    vals = signal.tobytes()

                    fields.append(frombuffer(vals, dtype="V6"))
                    types.append(("", "V6"))

                else:
                    vals = []
                    for field in ("ms", "min", "hour", "day", "month", "year"):
                        vals.append(signal[field])
                    vals = fromarrays(vals).tobytes()

                    fields.append(frombuffer(vals, dtype="V7"))
                    types.append(("", "V7"))

                if invalidation_bytes_nr and invalidation_bits is not None:
                    inval_bits.append(invalidation_bits)

            elif sig_type == v4c.SIGNAL_TYPE_STRUCTURE_COMPOSITION:

                if invalidation_bytes_nr and invalidation_bits is not None:
                    inval_bits.append(invalidation_bits)

                fields.append(signal)
                types.append(("", signal.dtype))

            elif sig_type == v4c.SIGNAL_TYPE_ARRAY:
                names = signal.dtype.names

                samples = signal[names[0]]

                shape = samples.shape[1:]

                fields.append(samples)
                types.append(("", samples.dtype, shape))

                if invalidation_bytes_nr and invalidation_bits is not None:
                    inval_bits.append(invalidation_bits)

                for name in names[1:]:

                    samples = signal[name]
                    shape = samples.shape[1:]
                    fields.append(samples)
                    types.append(("", samples.dtype, shape))

                    if invalidation_bytes_nr and invalidation_bits is not None:
                        inval_bits.append(invalidation_bits)

            else:
                if self.compact_vlsd:
                    cur_offset = sum(blk.original_size for blk in gp.signal_data[i])

                    data = []
                    offsets = []
                    off = 0
                    if gp.channels[i].data_type == v4c.DATA_TYPE_STRING_UTF_16_LE:
                        for elem in signal:
                            offsets.append(off)
                            size = len(elem)
                            if size % 2:
                                size += 1
                                elem = elem + b"\0"
                            data.append(UINT32_p(size))
                            data.append(elem)
                            off += size + 4
                    else:
                        for elem in signal:
                            offsets.append(off)
                            size = len(elem)
                            data.append(UINT32_p(size))
                            data.append(elem)
                            off += size + 4

                    offsets = array(offsets, dtype=uint64)

                    stream.seek(0, 2)
                    addr = stream.tell()

                    data_size = off
                    if data_size:
                        info = SignalDataBlockInfo(
                            address=addr,
                            compressed_size=data_size,
                            original_size=data_size,
                            location=v4c.LOCATION_TEMPORARY_FILE,
                        )
                        gp.signal_data[i].append(info)
                        stream.write(b"".join(data))

                    offsets += cur_offset
                    fields.append(offsets)
                    types.append(("", uint64))

                else:
                    cur_offset = sum(blk.original_size for blk in gp.signal_data[i])

                    offsets = arange(len(signal), dtype=uint64) * (signal.itemsize + 4)

                    values = [full(len(signal), signal.itemsize, dtype=uint32), signal]

                    types_ = [("", uint32), ("", signal.dtype)]

                    values = fromarrays(values, dtype=types_)

                    stream.seek(0, 2)
                    addr = stream.tell()
                    block_size = len(values) * values.itemsize
                    if block_size:
                        info = SignalDataBlockInfo(
                            address=addr,
                            compressed_size=block_size,
                            original_size=block_size,
                            location=v4c.LOCATION_TEMPORARY_FILE,
                        )
                        gp.signal_data[i].append(info)
                        values.tofile(stream)

                    offsets += cur_offset
                    fields.append(offsets)
                    types.append(("", uint64))

                if invalidation_bytes_nr and invalidation_bits is not None:
                    inval_bits.append(invalidation_bits)

        if invalidation_bytes_nr:
            invalidation_bytes_nr = len(inval_bits)
            cycles_nr = len(inval_bits[0])

            for _ in range(8 - invalidation_bytes_nr % 8):
                inval_bits.append(zeros(cycles_nr, dtype=bool))

            inval_bits.reverse()

            invalidation_bytes_nr = len(inval_bits) // 8

            gp.channel_group.invalidation_bytes_nr = invalidation_bytes_nr

            inval_bits = fliplr(
                packbits(array(inval_bits).T).reshape(
                    (cycles_nr, invalidation_bytes_nr)
                )
            )

            if self.version < "4.20":
                fields.append(inval_bits)
                types.append(
                    ("invalidation_bytes", inval_bits.dtype, inval_bits.shape[1:])
                )

        samples = fromarrays(fields, dtype=types)

        del fields
        del types

        stream.seek(0, 2)
        addr = stream.tell()
        size = len(samples) * samples.itemsize

        if size:

            if self.version < "4.20":
                data = samples.tobytes()
                raw_size = len(data)
                data = lz_compress(data)
                size = len(data)
                stream.write(data)
                gp.data_blocks.append(
                    DataBlockInfo(
                        address=addr,
                        block_type=v4c.DZ_BLOCK_LZ,
                        original_size=raw_size,
                        compressed_size=size,
                        param=0,
                    )
                )

                gp.channel_group.cycles_nr += added_cycles
                self.virtual_groups[index].cycles_nr += added_cycles

            else:
                data = samples.tobytes()
                raw_size = len(data)
                data = lz_compress(data)
                size = len(data)
                stream.write(data)

                gp.data_blocks.append(
                    DataBlockInfo(
                        address=addr,
                        block_type=v4c.DT_BLOCK_LZ,
                        original_size=raw_size,
                        compressed_size=size,
                        param=0,
                    )
                )

                gp.channel_group.cycles_nr += added_cycles
                self.virtual_groups[index].cycles_nr += added_cycles

                if invalidation_bytes_nr:
                    addr = stream.tell()

                    data = inval_bits.tobytes()
                    raw_size = len(data)
                    data = lz_compress(data)
                    size = len(data)
                    stream.write(data)

                    gp.data_blocks[-1].invalidation_block(
                        InvalidationBlockInfo(
                            address=addr,
                            block_type=v4c.DT_BLOCK_LZ,
                            original_size=raw_size,
                            compressed_size=size,
                            param=None,
                        )
                    )

    def _extend_column_oriented(self, index, signals):
        """
        Extend a group with new samples. *signals* contains (values, invalidation_bits)
        pairs for each extended signal. The first pair is the master channel's pair, and the
        next pairs must respect the same order in which the signals were appended. The samples must have raw
        or physical values according to the *Signals* used for the initial append.

        Parameters
        ----------
        index : int
            group index
        signals : list
            list on (numpy.ndarray, numpy.ndarray) objects

        Examples
        --------
        >>> # case 1 conversion type None
        >>> s1 = np.array([1, 2, 3, 4, 5])
        >>> s2 = np.array([-1, -2, -3, -4, -5])
        >>> s3 = np.array([0.1, 0.04, 0.09, 0.16, 0.25])
        >>> t = np.array([0.001, 0.002, 0.003, 0.004, 0.005])
        >>> names = ['Positive', 'Negative', 'Float']
        >>> units = ['+', '-', '.f']
        >>> s1 = Signal(samples=s1, timestamps=t, unit='+', name='Positive')
        >>> s2 = Signal(samples=s2, timestamps=t, unit='-', name='Negative')
        >>> s3 = Signal(samples=s3, timestamps=t, unit='flts', name='Floats')
        >>> mdf = MDF4('new.mdf')
        >>> mdf.append([s1, s2, s3], comment='created by asammdf v1.1.0')
        >>> t = np.array([0.006, 0.007, 0.008, 0.009, 0.010])
        >>> # extend without invalidation bits
        >>> mdf2.extend(0, [(t, None), (s1, None), (s2, None), (s3, None)])
        >>> # some invaldiation btis
        >>> s1_inv = np.array([0,0,0,1,1], dtype=np.bool)
        >>> mdf2.extend(0, [(t, None), (s1.samples, None), (s2.samples, None), (s3.samples, None)])

        """
        gp = self.groups[index]
        if not signals:
            message = '"append" requires a non-empty list of Signal objects'
            raise MdfException(message)

        stream = self._tempfile
        stream.seek(0, 2)
        write = stream.write
        tell = stream.tell

        added_cycles = len(signals[0][0])

        self.virtual_groups[index].cycles_nr += added_cycles

        for i, (signal, invalidation_bits) in enumerate(signals):
            gp = self.groups[index + i]
            sig_type = gp.signal_types[0]

            # first add the signals in the simple signal list
            if sig_type == v4c.SIGNAL_TYPE_SCALAR:
                samples = signal

            elif sig_type == v4c.SIGNAL_TYPE_CANOPEN:
                names = signal.dtype.names

                if names == v4c.CANOPEN_TIME_FIELDS:
                    samples = signal

                else:
                    vals = []
                    for field in ("ms", "min", "hour", "day", "month", "year"):
                        vals.append(signal[field])
                    samples = fromarrays(vals)

            elif sig_type == v4c.SIGNAL_TYPE_STRUCTURE_COMPOSITION:

                samples = signal

            elif sig_type == v4c.SIGNAL_TYPE_ARRAY:
                samples = signal

            else:
                cur_offset = sum(blk.original_size for blk in gp.signal_data[0])

                offsets = arange(len(signal), dtype=uint64) * (signal.itemsize + 4)

                values = [full(len(signal), signal.itemsize, dtype=uint32), signal]

                types_ = [("", uint32), ("", signal.dtype)]

                values = fromarrays(values, dtype=types_)

                addr = tell()
                block_size = len(values) * values.itemsize
                if block_size:
                    info = SignalDataBlockInfo(
                        address=addr,
                        compressed_size=block_size,
                        original_size=block_size,
                        location=v4c.LOCATION_TEMPORARY_FILE,
                    )
                    gp.signal_data[i].append(info)
                    write(values.tobytes())

                offsets += cur_offset

                samples = offsets

            addr = tell()

            if added_cycles:
                data = samples.tobytes()
                raw_size = len(data)
                data = lz_compress(data)

                size = len(data)
                write(data)

                gp.data_blocks.append(
                    DataBlockInfo(
                        address=addr,
                        block_type=v4c.DZ_BLOCK_LZ,
                        original_size=raw_size,
                        compressed_size=size,
                        param=0,
                    )
                )

                gp.channel_group.cycles_nr += added_cycles

                if invalidation_bits is not None:
                    addr = tell()
                    data = invalidation_bits.tobytes()
                    raw_size = len(data)
                    data = lz_compress(data)
                    size = len(data)
                    write(data)

                    gp.data_blocks[-1].invalidation_block(
                        InvalidationBlockInfo(
                            address=addr,
                            block_type=v4c.DZ_BLOCK_LZ,
                            original_size=raw_size,
                            compressed_size=size,
                            param=None,
                        )
                    )

    def attach(
        self,
        data,
        file_name=None,
        hash_sum=None,
        comment=None,
        compression=True,
        mime=r"application/octet-stream",
        embedded=True,
        encrypted=False,
        encryption_function=None,
    ):
        """attach embedded attachment as application/octet-stream.

        Parameters
        ----------
        data : bytes
            data to be attached
        file_name : str
            string file name
        hash_sum : bytes
            md5 of the data
        comment : str
            attachment comment
        compression : bool
            use compression for embedded attachment data
        mime : str
            mime type string
        embedded : bool
            attachment is embedded in the file
        encryption_function : bool, default None
            function used to encrypt the data. The function should only take a single bytes object as
            argument and return the encrypted bytes object. This is only valid for embedded attachments

            .. versionadded:: 6.2.0

        Returns
        -------
        index : int
            new attachment index

        """

        if hash_sum is None:
            worker = md5()
            worker.update(data)
            hash_sum = worker.hexdigest()
        hash_sum_encrypted = hash_sum

        encryption_function = (
            encryption_function or self._encryption_function or encryption.encrypt
        )

        if hash_sum in self._attachments_cache:
            return self._attachments_cache[hash_sum]
        else:
            creator_index = len(self.file_history)
            fh = FileHistory()
            fh.comment = """<FHcomment>
<TX>Added new embedded attachment from {}</TX>
<tool_id>asammdf</tool_id>
<tool_vendor>asammdf</tool_vendor>
<tool_version>{}</tool_version>
</FHcomment>""".format(
                file_name if file_name else "bin.bin", __version__
            )

            self.file_history.append(fh)

            file_name = file_name or "bin.bin"

            encrypted = False
            if encrypted and encryption_function is not None:
                try:
                    data = encryption_function(data)

                    worker = md5()
                    worker.update(data)
                    hash_sum_encrypted = worker.hexdigest()

                    if hash_sum_encrypted in self._attachments_cache:
                        return self._attachments_cache[hash_sum_encrypted]

                    encrypted = True
                except:
                    encrypted = False
            else:
                encrypted = False

            at_block = AttachmentBlock(
                data=data,
                compression=compression,
                embedded=embedded,
                file_name=file_name,
            )
            at_block["creator_index"] = creator_index
            if encrypted:
                at_block.flags |= v4c.FLAG_AT_ENCRYPTED

            self.attachments.append(at_block)

            suffix = Path(file_name).suffix.lower().strip(".")
            if suffix == "a2l":
                mime = "application/A2L"
            else:
                mime = f"application/x-{suffix}"

            at_block.mime = mime
            at_block.comment = comment

            index = len(self.attachments) - 1
            self._attachments_cache[hash_sum] = index
            self._attachments_cache[hash_sum_encrypted] = index

            return index

    def close(self):
        """if the MDF was created with memory=False and new
        channels have been appended, then this must be called just before the
        object is not used anymore to clean-up the temporary file"""

        self._parent = None
        if self._tempfile is not None:
            self._tempfile.close()
        if not self._from_filelike and self._file is not None:
            self._file.close()

        if self._delete_on_close:
            try:
                Path(self.name).unlink()
            except:
                pass

        if self.original_name is not None:
            if Path(self.original_name).suffix.lower() in (".bz2", ".gzip", ".mf4z", ".zip"):
                try:
                    Path(self.name).unlink()
                except:
                    pass

        for gp in self.groups:
            gp.clear()
        self.groups.clear()
        self.header = None
        self.identification = None
        self.file_history.clear()
        self.channels_db.clear()
        self.masters_db.clear()
        self.attachments.clear()
        self._attachments_cache.clear()
        self.file_comment = None
        self.events.clear()

        self._ch_map.clear()
        self._master_channel_metadata.clear()
        self._invalidation_cache.clear()
        self._external_dbc_cache.clear()
        self._si_map.clear()
        self._file_si_map.clear()
        self._cc_map.clear()
        self._file_cc_map.clear()
        self._cg_map.clear()
        self._cn_data_map.clear()
        self._dbc_cache.clear()
        self.virtual_groups.clear()

    @lru_cache(maxsize=128)
    def extract_attachment(self, index=None, decryption_function=None):
        """extract attachment data by index. If it is an embedded attachment,
        then this method creates the new file according to the attachment file
        name information

        Parameters
        ----------
        index : int
            attachment index; default *None*

        decryption_function : bool, default None
            function used to decrypt the data. The function should only take a single bytes object as
            argument and return the decrypted bytes object. This is only valid for embedded attachments

            .. versionadded:: 6.2.0

        Returns
        -------
        data : (bytes, pathlib.Path)
            tuple of attachment data and path

        """
        if index is None:
            return b"", Path(""), md5().digest()

        attachment = self.attachments[index]

        current_path = Path.cwd()
        file_path = Path(attachment.file_name or "embedded")
        decryption_function = (
            decryption_function or self._decryption_function or encryption.decrypt
        )
        try:
            os.chdir(self.name.resolve().parent)

            flags = attachment.flags

            # for embedded attachments extract data and create new files
            if flags & v4c.FLAG_AT_EMBEDDED:
                data = attachment.extract()
                md5_worker = md5()
                md5_worker.update(data)
                md5_sum = md5_worker.digest()

                if (
                    attachment.flags & v4c.FLAG_AT_ENCRYPTED
                    and decryption_function is not None
                ):
                    try:
                        data = decryption_function(data)
                    except:
                        pass

            else:

                # for external attachments read the file and return the content
                if flags & v4c.FLAG_AT_MD5_VALID:
                    data = open(file_path, "rb").read()
                    file_path = Path(f"FROM_{file_path}")
                    md5_worker = md5()
                    md5_worker.update(data)
                    md5_sum = md5_worker.digest()
                    if attachment["md5_sum"] == md5_sum:
                        if attachment.mime.startswith("text"):
                            with open(file_path, "r") as f:
                                data = f.read()
                    else:
                        message = (
                            f'ATBLOCK md5sum="{attachment["md5_sum"]}" '
                            f"and external attachment data ({file_path}) "
                            f'md5sum="{md5_sum}"'
                        )
                        logger.warning(message)
                else:
                    if attachment.mime.startswith("text"):
                        mode = "r"
                    else:
                        mode = "rb"
                    with open(file_path, mode) as f:
                        file_path = Path(f"FROM_{file_path}")
                        data = f.read()
                        md5_worker = md5()
                        md5_worker.update(data)
                        md5_sum = md5_worker.digest()

                if (
                    attachment.flags & v4c.FLAG_AT_ENCRYPTED
                    and decryption_function is not None
                ):
                    try:
                        data = decryption_function(data)
                    except:
                        pass

        except Exception as err:
            os.chdir(current_path)
            message = f'Exception during attachment "{attachment.file_name}" extraction: {err!r}'
            logger.warning(message)
            data = b""
            md5_sum = md5().digest()
        finally:
            os.chdir(current_path)

        return data, file_path, md5_sum

    def get(
        self,
        name=None,
        group=None,
        index=None,
        raster=None,
        samples_only=False,
        data=None,
        raw=False,
        ignore_invalidation_bits=False,
        record_offset=0,
        record_count=None,
    ):
        """Gets channel samples. The raw data group samples are not loaded to
        memory so it is advised to use ``filter`` or ``select`` instead of
        performing several ``get`` calls.

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
            return channel samples without applying the conversion rule; default
            `False`
        ignore_invalidation_bits : bool
            option to ignore invalidation bits
        record_offset : int
            if *data=None* use this to select the record offset from which the
            group data should be loaded
        record_count : int
            number of records to read; default *None* and in this case all
            available records are used


        Returns
        -------
        res : (numpy.array, numpy.array) | Signal
            returns *Signal* if *samples_only*=*False* (default option),
            otherwise returns a (numpy.array, numpy.array) tuple of samples and
            invalidation bits. If invalidation bits are not used or if
            *ignore_invalidation_bits* if False, then the second item will be
            None.

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
        UserWarning: Multiple occurrences for channel "Sig". Using first occurrence from data group 4. Provide both "group" and "index" arguments to select another data group
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

        gp_nr, ch_nr = self._validate_channel_selection(name, group, index)

        grp = self.groups[gp_nr]

        # get the channel object
        channel = grp.channels[ch_nr]
        dependency_list = grp.channel_dependencies[ch_nr]

        master_is_required = not samples_only or raster

        if dependency_list:
            if not isinstance(dependency_list[0], ChannelArrayBlock):
                vals, timestamps, invalidation_bits, encoding = self._get_structure(
                    channel=channel,
                    group=grp,
                    group_index=gp_nr,
                    channel_index=ch_nr,
                    dependency_list=dependency_list,
                    raster=raster,
                    data=data,
                    ignore_invalidation_bits=ignore_invalidation_bits,
                    record_offset=record_offset,
                    record_count=record_count,
                    master_is_required=master_is_required,
                )
            else:
                vals, timestamps, invalidation_bits, encoding = self._get_array(
                    channel=channel,
                    group=grp,
                    group_index=gp_nr,
                    channel_index=ch_nr,
                    dependency_list=dependency_list,
                    raster=raster,
                    data=data,
                    ignore_invalidation_bits=ignore_invalidation_bits,
                    record_offset=record_offset,
                    record_count=record_count,
                    master_is_required=master_is_required,
                )

        else:
            vals, timestamps, invalidation_bits, encoding = self._get_scalar(
                channel=channel,
                group=grp,
                group_index=gp_nr,
                channel_index=ch_nr,
                dependency_list=dependency_list,
                raster=raster,
                data=data,
                ignore_invalidation_bits=ignore_invalidation_bits,
                record_offset=record_offset,
                record_count=record_count,
                master_is_required=master_is_required,
            )

        conversion = channel.conversion

        if not raw and conversion:
            vals = conversion.convert(vals)
            conversion = None

            if vals.dtype.kind == "S":
                encoding = "utf-8"

        if not vals.flags.owndata and self.copy_on_get:
            vals = vals.copy()

        if samples_only:
            res = vals, invalidation_bits
        else:
            # search for unit in conversion texts

            channel_type = channel.channel_type

            if name is None:
                name = channel.name

            unit = conversion and conversion.unit or channel.unit

            comment = channel.comment

            source = channel.source

            if source:
                source = Source.from_source(source)
            else:
                cg_source = grp.channel_group.acq_source
                if cg_source:
                    source = Source.from_source(cg_source)
                else:
                    source = None

            if channel.attachment is not None:
                attachment = self.extract_attachment(
                    channel.attachment,
                    decryption_function=self._decryption_function,
                )
            else:
                attachment = None

            master_metadata = self._master_channel_metadata.get(gp_nr, None)

            stream_sync = channel_type == v4c.CHANNEL_TYPE_SYNC

            try:
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
                    bit_count=channel.bit_count,
                    stream_sync=stream_sync,
                    invalidation_bits=invalidation_bits,
                    encoding=encoding,
                    group_index=gp_nr,
                    channel_index=ch_nr,
                )
            except:
                debug_channel(self, grp, channel, dependency_list)
                raise

        return res

    def _get_structure(
        self,
        channel,
        group,
        group_index,
        channel_index,
        dependency_list,
        raster,
        data,
        ignore_invalidation_bits,
        record_offset,
        record_count,
        master_is_required,
    ):
        grp = group
        gp_nr = group_index
        # get data group record
        parents, dtypes = group.parents, group.types
        if parents is None:
            parents, dtypes = self._prepare_record(grp)

        # get group data
        if data is None:
            data = self._load_data(
                grp, record_offset=record_offset, record_count=record_count
            )
        else:
            data = (data,)

        channel_invalidation_present = channel.flags & (
            v4c.FLAG_CN_ALL_INVALID | v4c.FLAG_CN_INVALIDATION_PRESENT
        )

        _dtype = dtype(channel.dtype_fmt)
        if _dtype.itemsize == channel.bit_count // 8:
            fast_path = True
            channel_values = []
            timestamps = []
            invalidation_bits = []

            byte_offset = channel.byte_offset
            record_size = (
                grp.channel_group.samples_byte_nr
                + grp.channel_group.invalidation_bytes_nr
            )

            count = 0
            for fragment in data:

                bts = fragment[0]
                types = [
                    ("", f"V{byte_offset}"),
                    ("vals", _dtype),
                    ("", f"V{record_size - _dtype.itemsize - byte_offset}"),
                ]

                channel_values.append(fromstring(bts, types)["vals"].copy())

                if master_is_required:
                    timestamps.append(self.get_master(gp_nr, fragment, one_piece=True))
                if channel_invalidation_present:
                    invalidation_bits.append(
                        self.get_invalidation_bits(gp_nr, channel, fragment)
                    )

                count += 1
        else:
            unique_names = UniqueDB()
            fast_path = False
            names = [
                unique_names.get_unique_name(grp.channels[ch_nr].name)
                for _, ch_nr in dependency_list
            ]

            channel_values = [[] for _ in dependency_list]
            timestamps = []
            invalidation_bits = []

            count = 0
            for fragment in data:
                for i, (dg_nr, ch_nr) in enumerate(dependency_list):
                    vals = self.get(
                        group=dg_nr,
                        index=ch_nr,
                        samples_only=True,
                        data=fragment,
                        ignore_invalidation_bits=ignore_invalidation_bits,
                        record_offset=record_offset,
                        record_count=record_count,
                        raw=True,
                    )[0]
                    channel_values[i].append(vals)
                if master_is_required:
                    timestamps.append(self.get_master(gp_nr, fragment, one_piece=True))
                if channel_invalidation_present:
                    invalidation_bits.append(
                        self.get_invalidation_bits(gp_nr, channel, fragment)
                    )

                count += 1

        if fast_path:
            total_size = sum(len(_) for _ in channel_values)
            shape = (total_size,) + channel_values[0].shape[1:]

            if count > 1:
                out = empty(shape, dtype=channel_values[0].dtype)
                vals = concatenate(channel_values, out=out)
            else:
                vals = channel_values[0]
        else:
            total_size = sum(len(_) for _ in channel_values[0])

            if count > 1:
                arrays = [
                    concatenate(
                        lst,
                        out=empty((total_size,) + lst[0].shape[1:], dtype=lst[0].dtype),
                    )
                    for lst in channel_values
                ]
            else:
                arrays = [lst[0] for lst in channel_values]
            types = [
                (name_, arr.dtype, arr.shape[1:]) for name_, arr in zip(names, arrays)
            ]
            types = dtype(types)

            vals = fromarrays(arrays, dtype=types)

        if master_is_required:
            if count > 1:
                out = empty(total_size, dtype=timestamps[0].dtype)
                timestamps = concatenate(timestamps, out=out)
            else:
                timestamps = timestamps[0]
        else:
            timestamps = None

        if channel_invalidation_present:
            if count > 1:
                out = empty(total_size, dtype=invalidation_bits[0].dtype)
                invalidation_bits = concatenate(invalidation_bits, out=out)
            else:
                invalidation_bits = invalidation_bits[0]
            if not ignore_invalidation_bits:
                vals = vals[nonzero(~invalidation_bits)[0]]
                if master_is_required:
                    timestamps = timestamps[nonzero(~invalidation_bits)[0]]
                invalidation_bits = None
        else:
            invalidation_bits = None

        if raster and len(timestamps) > 1:
            t = arange(timestamps[0], timestamps[-1], raster)

            vals = Signal(
                vals, timestamps, name="_", invalidation_bits=invalidation_bits
            ).interp(
                t,
                integer_interpolation_mode=self._integer_interpolation,
                float_interpolation_mode=self._float_interpolation,
            )

            vals, timestamps, invalidation_bits = (
                vals.samples,
                vals.timestamps,
                vals.invalidation_bits,
            )

        return vals, timestamps, invalidation_bits, None

    def _get_array(
        self,
        channel,
        group,
        group_index,
        channel_index,
        dependency_list,
        raster,
        data,
        ignore_invalidation_bits,
        record_offset,
        record_count,
        master_is_required,
    ):

        grp = group
        gp_nr = group_index
        ch_nr = channel_index
        # get data group record
        parents, dtypes = group.parents, group.types
        if parents is None:
            parents, dtypes = self._prepare_record(grp)

        # get group data
        if data is None:
            data = self._load_data(
                grp, record_offset=record_offset, record_count=record_count
            )
        else:
            data = (data,)

        dep = ca_block = dependency_list[0]
        shape = tuple(ca_block[f"dim_size_{i}"] for i in range(ca_block.dims))
        shape = tuple(dim for dim in shape if dim > 1)
        shape = shape or (1,)

        dim = 1
        for d in shape:
            dim *= d

        item_size = channel.bit_count // 8
        size = item_size * dim

        if group.uses_ld:
            record_size = group.channel_group.samples_byte_nr
        else:
            record_size = (
                group.channel_group.samples_byte_nr
                + group.channel_group.invalidation_bytes_nr
            )

        channel_dtype = get_fmt_v4(
            channel.data_type,
            channel.bit_count,
            channel.channel_type,
        )

        byte_offset = channel.byte_offset

        types = [
            ("", f"a{byte_offset}"),
            ("vals", channel_dtype, shape),
            ("", f"a{record_size - size - byte_offset}"),
        ]

        dtype_fmt = dtype(types)

        channel_invalidation_present = channel.flags & (
            v4c.FLAG_CN_ALL_INVALID | v4c.FLAG_CN_INVALIDATION_PRESENT
        )

        channel_group = grp.channel_group
        samples_size = (
            channel_group.samples_byte_nr + channel_group.invalidation_bytes_nr
        )

        channel_values = []
        timestamps = []
        invalidation_bits = []
        count = 0

        for fragment in data:

            arrays = []
            types = []

            data_bytes, offset, _count, invalidation_bytes = fragment

            cycles = len(data_bytes) // samples_size

            vals = frombuffer(data_bytes, dtype=dtype_fmt)["vals"]

            if dep.flags & v4c.FLAG_CA_INVERSE_LAYOUT:
                shape = vals.shape
                shape = (shape[0],) + shape[1:][::-1]
                vals = vals.reshape(shape)

                axes = (0,) + tuple(range(len(shape) - 1, 0, -1))
                vals = transpose(vals, axes=axes)

            cycles_nr = len(vals)

            for ca_block in dependency_list[:1]:
                dims_nr = ca_block.dims

                if ca_block.ca_type == v4c.CA_TYPE_SCALE_AXIS:
                    shape = (ca_block.dim_size_0,)
                    arrays.append(vals)
                    dtype_pair = channel.name, vals.dtype, shape
                    types.append(dtype_pair)

                elif ca_block.ca_type == v4c.CA_TYPE_LOOKUP:
                    shape = vals.shape[1:]
                    arrays.append(vals)
                    dtype_pair = channel.name, vals.dtype, shape
                    types.append(dtype_pair)

                    if ca_block.flags & v4c.FLAG_CA_FIXED_AXIS:
                        for i in range(dims_nr):
                            shape = (ca_block[f"dim_size_{i}"],)
                            axis = []
                            for j in range(shape[0]):
                                key = f"axis_{i}_value_{j}"
                                axis.append(ca_block[key])
                            axis = array([axis for _ in range(cycles_nr)], dtype=f'{shape}f8')
                            arrays.append(axis)
                            dtype_pair = (f"axis_{i}", axis.dtype, shape)
                            types.append(dtype_pair)
                    else:
                        for i in range(dims_nr):

                            axis = ca_block.axis_channels[i]
                            shape = (ca_block[f"dim_size_{i}"],)

                            if axis is None:
                                axisname = f"axis_{i}"
                                axis_values = array([arange(shape[0])] * cycles, dtype=f"({shape[0]},)f8")

                            else:
                                try:
                                    (ref_dg_nr, ref_ch_nr) = ca_block.axis_channels[i]
                                except:
                                    debug_channel(self, grp, channel, dependency_list)
                                    raise

                                axisname = (
                                    self.groups[ref_dg_nr].channels[ref_ch_nr].name
                                )

                                if ref_dg_nr == gp_nr:
                                    axis_values = self.get(
                                        group=ref_dg_nr,
                                        index=ref_ch_nr,
                                        samples_only=True,
                                        data=fragment,
                                        ignore_invalidation_bits=ignore_invalidation_bits,
                                        record_offset=record_offset,
                                        record_count=cycles,
                                        raw=True,
                                    )[0]
                                else:
                                    channel_group = grp.channel_group
                                    record_size = channel_group.samples_byte_nr
                                    record_size += channel_group.invalidation_bytes_nr
                                    start = offset // record_size
                                    end = start + len(data_bytes) // record_size + 1
                                    ref = self.get(
                                        group=ref_dg_nr,
                                        index=ref_ch_nr,
                                        samples_only=True,
                                        ignore_invalidation_bits=ignore_invalidation_bits,
                                        record_offset=record_offset,
                                        record_count=cycles,
                                        raw=True,
                                    )[0]
                                    axis_values = ref[start:end].copy()

                                axis_values = axis_values[axisname]
                                if len(axis_values) == 0 and cycles:
                                    axis_values = array([arange(shape[0])] * cycles)

                            arrays.append(axis_values)
                            dtype_pair = (axisname, axis_values.dtype, shape)
                            types.append(dtype_pair)

                elif ca_block.ca_type == v4c.CA_TYPE_ARRAY:
                    shape = vals.shape[1:]
                    arrays.append(vals)
                    dtype_pair = channel.name, vals.dtype, shape
                    types.append(dtype_pair)

            for ca_block in dependency_list[1:]:
                dims_nr = ca_block.dims

                if ca_block.flags & v4c.FLAG_CA_FIXED_AXIS:
                    for i in range(dims_nr):
                        shape = (ca_block[f"dim_size_{i}"],)
                        axis = []
                        for j in range(shape[0]):
                            key = f"axis_{i}_value_{j}"
                            axis.append(ca_block[key])

                        axis = array([axis for _ in range(cycles_nr)], dtype=f'{shape}f8')
                        arrays.append(axis)
                        types.append((f"axis_{i}", axis.dtype, shape))
                else:
                    for i in range(dims_nr):
                        axis = ca_block.axis_channels[i]
                        shape = (ca_block[f"dim_size_{i}"],)

                        if axis is None:
                            axisname = f"axis_{i}"
                            axis_values = array([arange(shape[0])] * cycles, dtype=f"({shape[0]},)f8")

                        else:
                            try:
                                ref_dg_nr, ref_ch_nr = ca_block.axis_channels[i]
                            except:
                                debug_channel(self, grp, channel, dependency_list)
                                raise

                            axisname = self.groups[ref_dg_nr].channels[ref_ch_nr].name

                            if ref_dg_nr == gp_nr:
                                axis_values = self.get(
                                    group=ref_dg_nr,
                                    index=ref_ch_nr,
                                    samples_only=True,
                                    data=fragment,
                                    ignore_invalidation_bits=ignore_invalidation_bits,
                                    record_offset=record_offset,
                                    record_count=cycles,
                                    raw=True,
                                )[0]
                            else:
                                channel_group = grp.channel_group
                                record_size = channel_group.samples_byte_nr
                                record_size += channel_group.invalidation_bytes_nr
                                start = offset // record_size
                                end = start + len(data_bytes) // record_size + 1
                                ref = self.get(
                                    group=ref_dg_nr,
                                    index=ref_ch_nr,
                                    samples_only=True,
                                    ignore_invalidation_bits=ignore_invalidation_bits,
                                    record_offset=record_offset,
                                    record_count=cycles,
                                    raw=True,
                                )[0]
                                axis_values = ref[start:end].copy()
                            axis_values = axis_values[axisname]
                            if len(axis_values) == 0 and cycles:
                                axis_values = array([arange(shape[0])] * cycles)

                        arrays.append(axis_values)
                        dtype_pair = (axisname, axis_values.dtype, shape)
                        types.append(dtype_pair)

            vals = fromarrays(arrays, dtype(types))

            if master_is_required:
                timestamps.append(self.get_master(gp_nr, fragment, one_piece=True))
            if channel_invalidation_present:
                invalidation_bits.append(
                    self.get_invalidation_bits(gp_nr, channel, fragment)
                )

            channel_values.append(vals)
            count += 1

        if count > 1:
            total_size = sum(len(_) for _ in channel_values)
            shape = (total_size,) + channel_values[0].shape[1:]

        if count > 1:
            out = empty(shape, dtype=channel_values[0].dtype)
            vals = concatenate(channel_values, out=out)
        elif count == 1:
            vals = channel_values[0]
        else:
            vals = []

        if master_is_required:
            if count > 1:
                out = empty(total_size, dtype=timestamps[0].dtype)
                timestamps = concatenate(timestamps, out=out)
            else:
                timestamps = timestamps[0]
        else:
            timestamps = None

        if channel_invalidation_present:
            if count > 1:
                out = empty(total_size, dtype=invalidation_bits[0].dtype)
                invalidation_bits = concatenate(invalidation_bits, out=out)
            else:
                invalidation_bits = invalidation_bits[0]
            if not ignore_invalidation_bits:
                vals = vals[nonzero(~invalidation_bits)[0]]
                if master_is_required:
                    timestamps = timestamps[nonzero(~invalidation_bits)[0]]
                invalidation_bits = None
        else:
            invalidation_bits = None

        if raster and len(timestamps) > 1:
            t = arange(timestamps[0], timestamps[-1], raster)

            vals = Signal(
                vals, timestamps, name="_", invalidation_bits=invalidation_bits
            ).interp(
                t,
                integer_interpolation_mode=self._integer_interpolation,
                float_interpolation_mode=self._float_interpolation,
            )

            vals, timestamps, invalidation_bits = (
                vals.samples,
                vals.timestamps,
                vals.invalidation_bits,
            )

        return vals, timestamps, invalidation_bits, None

    def _get_scalar(
        self,
        channel,
        group,
        group_index,
        channel_index,
        dependency_list,
        raster,
        data,
        ignore_invalidation_bits,
        record_offset,
        record_count,
        master_is_required,
    ):
        grp = group
        gp_nr = group_index
        ch_nr = channel_index
        # get data group record
        parents, dtypes = group.parents, group.types
        if parents is None:
            parents, dtypes = self._prepare_record(grp)

        # get group data
        if data is None:
            data = self._load_data(
                grp, record_offset=record_offset, record_count=record_count
            )
            one_piece = False
        else:
            data = (data,)
            one_piece = True

        channel_invalidation_present = channel.flags & (
            v4c.FLAG_CN_ALL_INVALID | v4c.FLAG_CN_INVALIDATION_PRESENT
        )

        data_type = channel.data_type
        channel_type = channel.channel_type
        bit_count = channel.bit_count

        encoding = None

        if channel.dtype_fmt.subdtype:
            channel_dtype = channel.dtype_fmt.subdtype[0]
        else:
            channel_dtype = channel.dtype_fmt

        # get channel values
        if channel_type in {
            v4c.CHANNEL_TYPE_VIRTUAL,
            v4c.CHANNEL_TYPE_VIRTUAL_MASTER,
        }:
            if not channel.dtype_fmt:
                channel.dtype_fmt = get_fmt_v4(data_type, 64)
            ch_dtype = dtype(channel.dtype_fmt)

            channel_values = []
            timestamps = []
            invalidation_bits = []

            channel_group = grp.channel_group
            record_size = channel_group.samples_byte_nr
            record_size += channel_group.invalidation_bytes_nr

            count = 0
            for fragment in data:
                data_bytes, offset, _count, invalidation_bytes = fragment
                offset = offset // record_size

                vals = arange(len(data_bytes) // record_size, dtype=ch_dtype)
                vals += offset

                if master_is_required:
                    timestamps.append(
                        self.get_master(
                            gp_nr,
                            fragment,
                            record_offset=offset,
                            record_count=_count,
                            one_piece=True,
                        )
                    )
                if channel_invalidation_present:
                    invalidation_bits.append(
                        self.get_invalidation_bits(gp_nr, channel, fragment)
                    )

                channel_values.append(vals)
                count += 1

            if count > 1:
                total_size = sum(len(_) for _ in channel_values)
                shape = (total_size,) + channel_values[0].shape[1:]

            if count > 1:
                out = empty(shape, dtype=channel_values[0].dtype)
                vals = concatenate(channel_values, out=out)
            elif count == 1:
                vals = channel_values[0]
            else:
                vals = []

            if master_is_required:
                if count > 1:
                    out = empty(total_size, dtype=timestamps[0].dtype)
                    timestamps = concatenate(timestamps, out=out)
                else:
                    timestamps = timestamps[0]

            if channel_invalidation_present:
                if count > 1:
                    out = empty(total_size, dtype=invalidation_bits[0].dtype)
                    invalidation_bits = concatenate(invalidation_bits, out=out)
                else:
                    invalidation_bits = invalidation_bits[0]
                if not ignore_invalidation_bits:
                    vals = vals[nonzero(~invalidation_bits)[0]]
                    if master_is_required:
                        timestamps = timestamps[nonzero(~invalidation_bits)[0]]
                    invalidation_bits = None
            else:
                invalidation_bits = None

            if raster and len(timestamps) > 1:
                num = float(float32((timestamps[-1] - timestamps[0]) / raster))
                if num.is_integer():
                    t = linspace(timestamps[0], timestamps[-1], int(num))
                else:
                    t = arange(timestamps[0], timestamps[-1], raster)

                vals = Signal(
                    vals, timestamps, name="_", invalidation_bits=invalidation_bits
                ).interp(
                    t,
                    integer_interpolation_mode=self._integer_interpolation,
                    float_interpolation_mode=self._float_interpolation,
                )

                vals, timestamps, invalidation_bits = (
                    vals.samples,
                    vals.timestamps,
                    vals.invalidation_bits,
                )

            if channel.conversion:
                vals = channel.conversion.convert(vals)

        else:
            record_size = grp.channel_group.samples_byte_nr

            if one_piece:

                fragment = data[0]
                data_bytes, record_start, record_count, invalidation_bytes = fragment

                try:
                    parent, bit_offset = parents[ch_nr]
                except KeyError:
                    parent, bit_offset = None, None

                if parent is not None:
                    if (
                        len(grp.channels) == 1
                        and channel.dtype_fmt.itemsize == record_size
                    ):
                        vals = frombuffer(data_bytes, dtype=channel.dtype_fmt)
                    else:
                        record = grp.record
                        if record is None:
                            record = fromstring(data_bytes, dtype=dtypes)

                        vals = record[parent]

                    dtype_ = vals.dtype
                    shape_ = vals.shape
                    size = dtype_.itemsize
                    for dim in shape_[1:]:
                        size *= dim

                    kind_ = dtype_.kind

                    vals_dtype = vals.dtype.kind
                    if kind_ == "b":
                        pass
                    elif len(shape_) > 1 and data_type not in (
                        v4c.DATA_TYPE_BYTEARRAY,
                        v4c.DATA_TYPE_MIME_SAMPLE,
                        v4c.DATA_TYPE_MIME_STREAM,
                    ):
                        vals = self._get_not_byte_aligned_data(data_bytes, grp, ch_nr)
                    elif vals_dtype not in "ui" and (
                        bit_offset or not bit_count == size * 8
                    ):
                        vals = self._get_not_byte_aligned_data(data_bytes, grp, ch_nr)
                    else:
                        dtype_ = vals.dtype
                        kind_ = dtype_.kind

                        if data_type in v4c.INT_TYPES:

                            if channel_dtype.byteorder == "|" and data_type in (
                                v4c.DATA_TYPE_SIGNED_MOTOROLA,
                                v4c.DATA_TYPE_UNSIGNED_MOTOROLA,
                            ):
                                view = f">u{vals.itemsize}"
                            else:
                                view = f"{channel_dtype.byteorder}u{vals.itemsize}"

                            if dtype(view) != vals.dtype:
                                vals = vals.view(view)

                            if bit_offset:
                                vals = vals >> bit_offset

                            if bit_count != size * 8:
                                if data_type in v4c.SIGNED_INT:
                                    vals = as_non_byte_sized_signed_int(vals, bit_count)
                                else:
                                    mask = (1 << bit_count) - 1
                                    vals = vals & mask
                            elif data_type in v4c.SIGNED_INT:
                                view = f"{channel_dtype.byteorder}i{vals.itemsize}"
                                if dtype(view) != vals.dtype:
                                    vals = vals.view(view)

                        else:
                            if bit_count != size * 8:
                                vals = self._get_not_byte_aligned_data(
                                    data_bytes, grp, ch_nr
                                )
                            else:
                                if kind_ in "ui":
                                    try:
                                        vals = vals.view(channel_dtype)
                                    except ValueError:
                                        vals = vals.copy().view(channel_dtype)

                else:
                    vals = self._get_not_byte_aligned_data(data_bytes, grp, ch_nr)

                if self._single_bit_uint_as_bool and bit_count == 1:
                    vals = array(vals, dtype=bool)
                else:
                    if vals.dtype != channel_dtype:
                        try:
                            vals = vals.astype(channel_dtype)
                        except ValueError:
                            vals = vals.copy().view(channel_dtype)

                if master_is_required:
                    timestamps = self.get_master(gp_nr, fragment, one_piece=True)
                else:
                    timestamps = None

                if channel_invalidation_present:
                    invalidation_bits = self.get_invalidation_bits(
                        gp_nr, channel, fragment
                    )

                    if not ignore_invalidation_bits:
                        vals = vals[nonzero(~invalidation_bits)[0]]
                        if master_is_required:
                            timestamps = timestamps[nonzero(~invalidation_bits)[0]]
                        invalidation_bits = None
                else:
                    invalidation_bits = None
            else:
                channel_values = []
                timestamps = []
                invalidation_bits = []

                for count, fragment in enumerate(data, 1):
                    data_bytes, offset, _count, invalidation_bytes = fragment

                    if count == 1:
                        record_start = offset
                        record_count = _count
                    try:
                        parent, bit_offset = parents[ch_nr]
                    except KeyError:
                        parent, bit_offset = None, None

                    if parent is not None:
                        if (
                            len(grp.channels) == 1
                            and channel.dtype_fmt.itemsize == record_size
                        ):
                            vals = frombuffer(data_bytes, dtype=channel.dtype_fmt)
                        else:
                            record = grp.record
                            if record is None:
                                record = fromstring(data_bytes, dtype=dtypes)

                            vals = record[parent]

                        dtype_ = vals.dtype
                        shape_ = vals.shape
                        size = dtype_.itemsize
                        for dim in shape_[1:]:
                            size *= dim

                        kind_ = dtype_.kind

                        vals_dtype = vals.dtype.kind
                        if kind_ == "b":
                            pass
                        elif len(shape_) > 1 and data_type not in (
                            v4c.DATA_TYPE_BYTEARRAY,
                            v4c.DATA_TYPE_MIME_SAMPLE,
                            v4c.DATA_TYPE_MIME_STREAM,
                        ):
                            vals = self._get_not_byte_aligned_data(
                                data_bytes, grp, ch_nr
                            )
                        elif vals_dtype not in "ui" and (
                            bit_offset or not bit_count == size * 8
                        ):
                            vals = self._get_not_byte_aligned_data(
                                data_bytes, grp, ch_nr
                            )
                        else:
                            dtype_ = vals.dtype
                            kind_ = dtype_.kind

                            if data_type in v4c.INT_TYPES:

                                if channel_dtype.byteorder == "|" and data_type in (
                                    v4c.DATA_TYPE_SIGNED_MOTOROLA,
                                    v4c.DATA_TYPE_UNSIGNED_MOTOROLA,
                                ):
                                    view = f">u{vals.itemsize}"
                                else:
                                    view = f"{channel_dtype.byteorder}u{vals.itemsize}"

                                if dtype(view) != vals.dtype:
                                    vals = vals.view(view)

                                if bit_offset:
                                    vals = vals >> bit_offset

                                if bit_count != size * 8:
                                    if data_type in v4c.SIGNED_INT:
                                        vals = as_non_byte_sized_signed_int(
                                            vals, bit_count
                                        )
                                    else:
                                        mask = (1 << bit_count) - 1
                                        vals = vals & mask
                                elif data_type in v4c.SIGNED_INT:
                                    view = f"{channel_dtype.byteorder}i{vals.itemsize}"
                                    if dtype(view) != vals.dtype:
                                        vals = vals.view(view)

                            else:
                                if bit_count != size * 8:
                                    vals = self._get_not_byte_aligned_data(
                                        data_bytes, grp, ch_nr
                                    )
                                else:
                                    if kind_ in "ui":
                                        vals = vals.view(channel_dtype)

                    else:
                        vals = self._get_not_byte_aligned_data(data_bytes, grp, ch_nr)

                    if bit_count == 1 and self._single_bit_uint_as_bool:
                        vals = array(vals, dtype=bool)
                    else:

                        if vals.dtype != channel_dtype:
                            vals = vals.astype(channel_dtype)

                    if master_is_required:
                        timestamps.append(
                            self.get_master(gp_nr, fragment, one_piece=True)
                        )
                    if channel_invalidation_present:
                        invalidation_bits.append(
                            self.get_invalidation_bits(gp_nr, channel, fragment)
                        )

                    if vals.flags.owndata:
                        channel_values.append(vals)
                    else:
                        channel_values.append(vals.copy())

                if count > 1:
                    total_size = sum(len(_) for _ in channel_values)
                    shape = (total_size,) + channel_values[0].shape[1:]

                    out = empty(shape, dtype=channel_values[0].dtype)
                    vals = concatenate(channel_values, out=out)
                elif count == 1:
                    vals = channel_values[0]
                else:
                    vals = array([], dtype=channel.dtype_fmt)

                if master_is_required:
                    if count > 1:
                        out = empty(total_size, dtype=timestamps[0].dtype)
                        timestamps = concatenate(timestamps, out=out)
                    elif count == 1:
                        timestamps = timestamps[0]
                    else:
                        timestamps = []

                if channel_invalidation_present:
                    if count > 1:
                        out = empty(total_size, dtype=invalidation_bits[0].dtype)
                        invalidation_bits = concatenate(invalidation_bits, out=out)
                    elif count == 1:
                        invalidation_bits = invalidation_bits[0]
                    else:
                        invalidation_bits = []
                    if not ignore_invalidation_bits:
                        vals = vals[nonzero(~invalidation_bits)[0]]
                        if master_is_required:
                            timestamps = timestamps[nonzero(~invalidation_bits)[0]]
                        invalidation_bits = None
                else:
                    invalidation_bits = None

            if raster and len(timestamps) > 1:

                num = float(float32((timestamps[-1] - timestamps[0]) / raster))
                if num.is_integer():
                    t = linspace(timestamps[0], timestamps[-1], int(num))
                else:
                    t = arange(timestamps[0], timestamps[-1], raster)

                vals = Signal(
                    vals, timestamps, name="_", invalidation_bits=invalidation_bits
                ).interp(
                    t,
                    integer_interpolation_mode=self._integer_interpolation,
                    float_interpolation_mode=self._float_interpolation,
                )

                vals, timestamps, invalidation_bits = (
                    vals.samples,
                    vals.timestamps,
                    vals.invalidation_bits,
                )

        if channel_type == v4c.CHANNEL_TYPE_VLSD:
            count_ = len(vals)

            if count_:
                signal_data = self._load_signal_data(
                    group=grp, index=ch_nr, start_offset=vals[0], end_offset=vals[-1]
                )
            else:
                signal_data = b""

            if signal_data:
                if data_type in (
                    v4c.DATA_TYPE_BYTEARRAY,
                    v4c.DATA_TYPE_UNSIGNED_INTEL,
                    v4c.DATA_TYPE_UNSIGNED_MOTOROLA,
                ):
                    vals = extract(signal_data, 1)
                else:
                    vals = extract(signal_data, 0)

                if data_type not in (
                    v4c.DATA_TYPE_BYTEARRAY,
                    v4c.DATA_TYPE_UNSIGNED_INTEL,
                    v4c.DATA_TYPE_UNSIGNED_MOTOROLA,
                ):

                    if data_type == v4c.DATA_TYPE_STRING_UTF_16_BE:
                        encoding = "utf-16-be"

                    elif data_type == v4c.DATA_TYPE_STRING_UTF_16_LE:
                        encoding = "utf-16-le"

                    elif data_type == v4c.DATA_TYPE_STRING_UTF_8:
                        encoding = "utf-8"

                    elif data_type == v4c.DATA_TYPE_STRING_LATIN_1:
                        encoding = "latin-1"

                    else:
                        raise MdfException(
                            f'wrong data type "{data_type}" for vlsd channel'
                        )
            else:
                if len(vals):
                    raise MdfException(
                        f'Wrong signal data block refence (0x{channel.data_block_addr:X}) for VLSD channel "{channel.name}"'
                    )
                # no VLSD signal data samples
                if data_type != v4c.DATA_TYPE_BYTEARRAY:
                    vals = array([], dtype="S")

                    if data_type == v4c.DATA_TYPE_STRING_UTF_16_BE:
                        encoding = "utf-16-be"

                    elif data_type == v4c.DATA_TYPE_STRING_UTF_16_LE:
                        encoding = "utf-16-le"

                    elif data_type == v4c.DATA_TYPE_STRING_UTF_8:
                        encoding = "utf-8"

                    elif data_type == v4c.DATA_TYPE_STRING_LATIN_1:
                        encoding = "latin-1"

                    else:
                        raise MdfException(
                            f'wrong data type "{data_type}" for vlsd channel'
                        )
                else:
                    vals = array(
                        [],
                        dtype=get_fmt_v4(data_type, bit_count, v4c.CHANNEL_TYPE_VALUE),
                    )

        elif not (
            v4c.DATA_TYPE_STRING_LATIN_1 <= data_type <= v4c.DATA_TYPE_STRING_UTF_16_BE
        ):
            pass

        elif channel_type in {v4c.CHANNEL_TYPE_VALUE, v4c.CHANNEL_TYPE_MLSD}:

            if data_type == v4c.DATA_TYPE_STRING_UTF_16_BE:
                encoding = "utf-16-be"

            elif data_type == v4c.DATA_TYPE_STRING_UTF_16_LE:
                encoding = "utf-16-le"

            elif data_type == v4c.DATA_TYPE_STRING_UTF_8:
                encoding = "utf-8"

            elif data_type == v4c.DATA_TYPE_STRING_LATIN_1:
                encoding = "latin-1"

            else:
                raise MdfException(f'wrong data type "{data_type}" for string channel')

        if (
            data_type < v4c.DATA_TYPE_CANOPEN_DATE
            or data_type > v4c.DATA_TYPE_CANOPEN_TIME
        ):
            pass
        else:
            # CANopen date
            if data_type == v4c.DATA_TYPE_CANOPEN_DATE:

                types = dtype(
                    [
                        ("ms", "<u2"),
                        ("min", "<u1"),
                        ("hour", "<u1"),
                        ("day", "<u1"),
                        ("month", "<u1"),
                        ("year", "<u1"),
                    ]
                )
                vals = vals.view(types)

                arrays = []
                arrays.append(vals["ms"])
                # bit 6 and 7 of minutes are reserved
                arrays.append(vals["min"] & 0x3F)
                # only firt 4 bits of hour are used
                arrays.append(vals["hour"] & 0xF)
                # the first 4 bits are the day number
                arrays.append(vals["day"] & 0xF)
                # bit 6 and 7 of month are reserved
                arrays.append(vals["month"] & 0x3F)
                # bit 7 of year is reserved
                arrays.append(vals["year"] & 0x7F)
                # add summer or standard time information for hour
                arrays.append((vals["hour"] & 0x80) >> 7)
                # add day of week information
                arrays.append((vals["day"] & 0xF0) >> 4)

                names = [
                    "ms",
                    "min",
                    "hour",
                    "day",
                    "month",
                    "year",
                    "summer_time",
                    "day_of_week",
                ]
                vals = fromarrays(arrays, names=names)

            # CANopen time
            elif data_type == v4c.DATA_TYPE_CANOPEN_TIME:

                types = dtype([("ms", "<u4"), ("days", "<u2")])
                vals = vals.view(types)

                arrays = []
                # bits 28 to 31 are reserverd for ms
                arrays.append(vals["ms"] & 0xFFFFFFF)
                arrays.append(vals["days"] & 0x3F)

                names = ["ms", "days"]
                vals = fromarrays(arrays, names=names)

        return vals, timestamps, invalidation_bits, encoding

    def _get_not_byte_aligned_data(self, data, group, ch_nr):
        big_endian_types = (
            v4c.DATA_TYPE_UNSIGNED_MOTOROLA,
            v4c.DATA_TYPE_REAL_MOTOROLA,
            v4c.DATA_TYPE_SIGNED_MOTOROLA,
        )

        if group.uses_ld:
            record_size = group.channel_group.samples_byte_nr
        else:
            record_size = (
                group.channel_group.samples_byte_nr
                + group.channel_group.invalidation_bytes_nr
            )

        channel = group.channels[ch_nr]

        bit_offset = channel.bit_offset
        byte_offset = channel.byte_offset
        bit_count = channel.bit_count

        if ch_nr >= 0:
            dependencies = group.channel_dependencies[ch_nr]
            if dependencies and isinstance(dependencies[0], ChannelArrayBlock):
                ca_block = dependencies[0]

                size = bit_count // 8

                shape = tuple(ca_block[f"dim_size_{i}"] for i in range(ca_block.dims))
                if ca_block.byte_offset_base // size > 1 and len(shape) == 1:
                    shape += (ca_block.byte_offset_base // size,)
                dim = 1
                for d in shape:
                    dim *= d
                size *= dim
                bit_count = size * 8

        byte_size = bit_offset + bit_count
        if byte_size % 8:
            byte_size = (byte_size // 8) + 1
        else:
            byte_size //= 8

        types = [
            ("", f"a{byte_offset}"),
            ("vals", f"({byte_size},)u1"),
            ("", f"a{record_size - byte_size - byte_offset}"),
        ]

        vals = fromstring(data, dtype=dtype(types))

        vals = vals["vals"]

        if byte_size in {1, 2, 4, 8}:
            extra_bytes = 0
        elif byte_size < 8:
            extra_bytes = 4 - (byte_size % 4)
        else:
            extra_bytes = 0

        std_size = byte_size + extra_bytes

        big_endian = channel.data_type in big_endian_types

        # prepend or append extra bytes columns
        # to get a standard size number of bytes

        if extra_bytes:
            if big_endian:

                vals = column_stack(
                    [vals, zeros(len(vals), dtype=f"<({extra_bytes},)u1")]
                )
                try:
                    vals = vals.view(f">u{std_size}").ravel()
                except:
                    vals = frombuffer(vals.tobytes(), dtype=f">u{std_size}")

                vals = vals >> (extra_bytes * 8 + bit_offset)
                vals &= (1 << bit_count) - 1

            else:
                vals = column_stack(
                    [vals, zeros(len(vals), dtype=f"<({extra_bytes},)u1")]
                )
                try:
                    vals = vals.view(f"<u{std_size}").ravel()
                except:
                    vals = frombuffer(vals.tobytes(), dtype=f"<u{std_size}")

                vals = vals >> bit_offset
                vals &= (1 << bit_count) - 1

        else:
            if big_endian:
                try:
                    vals = vals.view(f">u{std_size}").ravel()
                except:
                    vals = frombuffer(vals.tobytes(), dtype=f">u{std_size}")

                vals = vals >> bit_offset
                vals &= (1 << bit_count) - 1

            else:
                try:
                    vals = vals.view(f"<u{std_size}").ravel()
                except:
                    vals = frombuffer(vals.tobytes(), dtype=f"<u{std_size}")

                vals = vals >> bit_offset
                vals &= (1 << bit_count) - 1

        data_type = channel.data_type

        if data_type in v4c.SIGNED_INT:
            return as_non_byte_sized_signed_int(vals, bit_count)
        elif data_type in v4c.FLOATS:
            return vals.view(get_fmt_v4(data_type, bit_count))
        else:
            return vals

    def included_channels(
        self, index=None, channels=None, skip_master=True, minimal=True
    ):

        if channels is None:
            virtual_channel_group = self.virtual_groups[index]
            groups = virtual_channel_group.groups

            gps = {}

            for gp_index in groups:

                group = self.groups[gp_index]

                included_channels = set(range(len(group.channels)))
                master_index = self.masters_db.get(gp_index, None)
                if master_index is not None:
                    included_channels.remove(master_index)

                channels = group.channels

                for dependencies in group.channel_dependencies:
                    if dependencies is None:
                        continue

                    if all(
                        not isinstance(dep, ChannelArrayBlock) for dep in dependencies
                    ):

                        for _, ch_nr in dependencies:
                            try:
                                included_channels.remove(ch_nr)
                            except KeyError:
                                pass
                    else:
                        for dep in dependencies:
                            for referenced_channels in (
                                dep.axis_channels,
                                dep.dynamic_size_channels,
                                dep.input_quantity_channels,
                            ):
                                for gp_nr, ch_nr in referenced_channels:
                                    if gp_nr == gp_index:
                                        try:
                                            included_channels.remove(ch_nr)
                                        except KeyError:
                                            pass

                            if dep.output_quantity_channel:
                                gp_nr, ch_nr = dep.output_quantity_channel
                                if gp_nr == gp_index:
                                    try:
                                        included_channels.remove(ch_nr)
                                    except KeyError:
                                        pass

                            if dep.comparison_quantity_channel:
                                gp_nr, ch_nr = dep.comparison_quantity_channel
                                if gp_nr == gp_index:
                                    try:
                                        included_channels.remove(ch_nr)
                                    except KeyError:
                                        pass

                gps[gp_index] = sorted(included_channels)

            result = {index: gps}
        else:
            gps = {}
            for item in channels:
                if isinstance(item, (list, tuple)):
                    if len(item) not in (2, 3):
                        raise MdfException(
                            "The items used for filtering must be strings, "
                            "or they must match the first 3 arguments of the get "
                            "method"
                        )
                    else:
                        group, idx = self._validate_channel_selection(*item)
                        gps_idx = gps.setdefault(group, set())
                        gps_idx.add(idx)
                else:
                    name = item
                    group, idx = self._validate_channel_selection(name)
                    gps_idx = gps.setdefault(group, set())
                    gps_idx.add(idx)

            result = {}
            for gp_index, channels in gps.items():
                master = self.virtual_groups_map[gp_index]
                group = self.groups[gp_index]

                if minimal:

                    channel_dependencies = [
                        group.channel_dependencies[ch_nr] for ch_nr in channels
                    ]

                    for dependencies in channel_dependencies:
                        if dependencies is None:
                            continue

                        if all(
                            not isinstance(dep, ChannelArrayBlock)
                            for dep in dependencies
                        ):

                            for _, ch_nr in dependencies:
                                try:
                                    channels.remove(ch_nr)
                                except KeyError:
                                    pass
                        else:
                            for dep in dependencies:
                                for referenced_channels in (
                                    dep.axis_channels,
                                    dep.dynamic_size_channels,
                                    dep.input_quantity_channels,
                                ):
                                    for gp_nr, ch_nr in referenced_channels:
                                        if gp_nr == gp_index:
                                            try:
                                                channels.remove(ch_nr)
                                            except KeyError:
                                                pass

                                if dep.output_quantity_channel:
                                    gp_nr, ch_nr = dep.output_quantity_channel
                                    if gp_nr == gp_index:
                                        try:
                                            channels.remove(ch_nr)
                                        except KeyError:
                                            pass

                                if dep.comparison_quantity_channel:
                                    gp_nr, ch_nr = dep.comparison_quantity_channel
                                    if gp_nr == gp_index:
                                        try:
                                            channels.remove(ch_nr)
                                        except KeyError:
                                            pass

                    gp_master = self.masters_db.get(gp_index, None)
                    if gp_master is not None and gp_master in channels:
                        channels.remove(gp_master)

                if master not in result:
                    result[master] = {}
                    result[master][master] = [self.masters_db.get(master, None)]

                result[master][gp_index] = sorted(channels)

        return result

    def _yield_selected_signals(
        self,
        index,
        groups=None,
        record_offset=0,
        record_count=None,
        skip_master=True,
        version=None,
    ):
        version = version or self.version
        virtual_channel_group = self.virtual_groups[index]
        record_size = virtual_channel_group.record_size

        if groups is None:
            groups = self.included_channels(index, skip_master=skip_master)[index]

        record_size = 0
        for group_index in groups:
            grp = self.groups[group_index]
            record_size += (
                grp.channel_group.samples_byte_nr
                + grp.channel_group.invalidation_bytes_nr
            )

        record_size = record_size or 1

        if self._read_fragment_size:
            count = self._read_fragment_size // record_size or 1
        else:
            if version < "4.20":
                count = 16 * 1024 * 1024 // record_size or 1
            else:
                count = 128 * 1024 * 1024 // record_size or 1

        data_streams = []
        for idx, group_index in enumerate(groups):
            grp = self.groups[group_index]
            grp.read_split_count = count
            data_streams.append(
                self._load_data(
                    grp, record_offset=record_offset, record_count=record_count
                )
            )
            if group_index == index:
                master_index = idx

        encodings = {group_index: [None] for groups_index in groups}

        self._set_temporary_master(None)
        idx = 0

        while True:
            try:
                fragments = [next(stream) for stream in data_streams]
            except:
                break

            _master = self.get_master(index, data=fragments[master_index])
            self._set_temporary_master(_master)

            if idx == 0:
                signals = []
            else:
                signals = [(_master, None)]

            for fragment, (group_index, channels) in zip(fragments, groups.items()):
                grp = self.groups[group_index]
                if not grp.single_channel_dtype:
                    parents, dtypes = self._prepare_record(grp)
                    if dtypes.itemsize:
                        grp.record = fromstring(fragment[0], dtype=dtypes)
                    else:
                        grp.record = None
                        continue

                if idx == 0:
                    for channel_index in channels:
                        signals.append(
                            self.get(
                                group=group_index,
                                index=channel_index,
                                data=fragment,
                                raw=True,
                                ignore_invalidation_bits=True,
                                samples_only=False,
                            )
                        )

                else:
                    for channel_index in channels:
                        signals.append(
                            self.get(
                                group=group_index,
                                index=channel_index,
                                data=fragment,
                                raw=True,
                                ignore_invalidation_bits=True,
                                samples_only=True,
                            )
                        )

                if version < "4.00":
                    if idx == 0:
                        for sig, channel_index in zip(signals, channels):
                            if sig.samples.dtype.kind == "S":

                                strsig = self.get(
                                    group=group_index,
                                    index=channel_index,
                                    samples_only=True,
                                    ignore_invalidation_bits=True,
                                )[0]

                                _dtype = strsig.dtype
                                sig.samples = sig.samples.astype(_dtype)
                                encodings[group_index].append((sig.encoding, _dtype))
                                del strsig
                                if sig.encoding != "latin-1":

                                    if sig.encoding == "utf-16-le":
                                        sig.samples = (
                                            sig.samples.view(uint16)
                                            .byteswap()
                                            .view(sig.samples.dtype)
                                        )
                                        sig.samples = encode(
                                            decode(sig.samples, "utf-16-be"), "latin-1"
                                        )
                                    else:
                                        sig.samples = encode(
                                            decode(sig.samples, sig.encoding),
                                            "latin-1",
                                        )
                                sig.samples = sig.samples.astype(_dtype)
                            else:
                                encodings[group_index].append(None)
                    else:
                        for i, (sig, encoding_tuple) in enumerate(
                            zip(signals, encodings[group_index])
                        ):

                            if encoding_tuple:
                                encoding, _dtype = encoding_tuple
                                samples = sig[0]
                                if encoding != "latin-1":

                                    if encoding == "utf-16-le":
                                        samples = (
                                            samples.view(uint16)
                                            .byteswap()
                                            .view(samples.dtype)
                                        )
                                        samples = encode(
                                            decode(samples, "utf-16-be"), "latin-1"
                                        )
                                    else:
                                        samples = encode(
                                            decode(samples, encoding), "latin-1"
                                        )
                                samples = samples.astype(_dtype)
                                signals[i] = (samples, sig[1])

                grp.record = None

            self._set_temporary_master(None)
            idx += 1
            yield signals

    def get_master(
        self,
        index,
        data=None,
        raster=None,
        record_offset=0,
        record_count=None,
        one_piece=False,
    ):
        """returns master channel samples for given group

        Parameters
        ----------
        index : int
            group index
        data : (bytes, int, int, bytes|None)
            (data block raw bytes, fragment offset, count, invalidation bytes); default None
        raster : float
            raster to be used for interpolation; default None

            .. deprecated:: 5.13.0

        record_offset : int
            if *data=None* use this to select the record offset from which the
            group data should be loaded
        record_count : int
            number of records to read; default *None* and in this case all
            available records are used

        Returns
        -------
        t : numpy.array
            master channel samples

        """

        if raster is not None:
            PendingDeprecationWarning(
                "the argument raster is deprecated since version 5.13.0 "
                "and will be removed in a future release"
            )
        if self._master is not None:
            return self._master

        group = self.groups[index]
        if group.channel_group.flags & v4c.FLAG_CG_REMOTE_MASTER:
            if data is not None:
                record_offset = data[1]
                record_count = data[2]
            return self.get_master(
                group.channel_group.cg_master_index,
                record_offset=record_offset,
                record_count=record_count,
            )

        time_ch_nr = self.masters_db.get(index, None)
        channel_group = group.channel_group
        record_size = channel_group.samples_byte_nr
        record_size += channel_group.invalidation_bytes_nr
        if record_count is not None:
            cycles_nr = record_count
        else:
            cycles_nr = group.channel_group.cycles_nr

        fragment = data
        if fragment:
            data_bytes, offset, _count, invalidation_bytes = fragment
            cycles_nr = len(data_bytes) // record_size if record_size else 0
        else:
            offset = 0
            _count = record_count

        if time_ch_nr is None:
            if record_size:
                t = arange(cycles_nr, dtype=float64)
                t += offset
            else:
                t = array([], dtype=float64)
            metadata = ("timestamps", v4c.SYNC_TYPE_TIME)
        else:

            time_ch = group.channels[time_ch_nr]
            time_conv = time_ch.conversion
            time_name = time_ch.name

            metadata = (time_name, time_ch.sync_type)

            if time_ch.channel_type == v4c.CHANNEL_TYPE_VIRTUAL_MASTER:
                time_a = time_conv["a"]
                time_b = time_conv["b"]
                t = arange(cycles_nr, dtype=float64)
                t += offset
                t *= time_a
                t += time_b

                if record_count is None:
                    t = t[record_offset:]
                else:
                    t = t[record_offset : record_offset + record_count]

            else:
                # check if the channel group contains just the master channel
                # and that there are no padding bytes
                if (
                    len(group.channels) == 1
                    and time_ch.dtype_fmt.itemsize == record_size
                ):
                    if one_piece:

                        data_bytes, offset, _count, _ = data

                        t = frombuffer(data_bytes, dtype=time_ch.dtype_fmt)
                    else:

                        # get data
                        if fragment is None:
                            data = self._load_data(
                                group,
                                record_offset=record_offset,
                                record_count=record_count,
                            )
                        else:
                            data = (fragment,)

                        time_values = [
                            frombuffer(fragment[0], dtype=time_ch.dtype_fmt)
                            for fragment in data
                        ]

                        if len(time_values) > 1:
                            total_size = sum(len(_) for _ in time_values)

                            out = empty(total_size, dtype=time_ch.dtype_fmt)
                            t = concatenate(time_values, out=out)
                        else:
                            t = time_values[0]

                else:
                    # get data group parents and dtypes
                    parents, dtypes = group.parents, group.types
                    if parents is None:
                        parents, dtypes = self._prepare_record(group)

                    if one_piece:
                        data_bytes, offset, _count, _ = data
                        try:
                            parent, _ = parents[time_ch_nr]
                        except KeyError:
                            parent = None
                        if parent is not None:
                            if group.record is None:
                                dtypes = group.types
                                if dtypes.itemsize:
                                    record = fromstring(data_bytes, dtype=dtypes)
                                else:
                                    record = None
                            else:
                                record = group.record

                            t = record[parent].copy()
                        else:
                            t = self._get_not_byte_aligned_data(
                                data_bytes, group, time_ch_nr
                            )

                    else:

                        # get data
                        if fragment is None:
                            data = self._load_data(
                                group,
                                record_offset=record_offset,
                                record_count=record_count,
                            )
                        else:
                            data = (fragment,)

                        time_values = []

                        count = 0

                        for fragment in data:
                            data_bytes, offset, _count, invalidation_bytes = fragment
                            try:
                                parent, _ = parents[time_ch_nr]
                            except KeyError:
                                parent = None
                            if parent is not None:
                                if group.record is None:
                                    dtypes = group.types
                                    if dtypes.itemsize:
                                        record = fromstring(data_bytes, dtype=dtypes)
                                    else:
                                        record = None
                                else:
                                    record = group.record

                                t = record[parent].copy()
                            else:
                                t = self._get_not_byte_aligned_data(
                                    data_bytes, group, time_ch_nr
                                )

                            time_values.append(t)
                            count += 1

                        if count > 1:
                            total_size = sum(len(_) for _ in time_values)

                        if len(time_values) > 1:
                            out = empty(total_size, dtype=time_values[0].dtype)
                            t = concatenate(time_values, out=out)
                        else:
                            t = time_values[0]

                # get timestamps
                if time_conv:
                    t = time_conv.convert(t)

        self._master_channel_metadata[index] = metadata

        if not t.dtype == float64:
            t = t.astype(float64)

        if raster and t.size:
            timestamps = t
            if len(t) > 1:
                num = float(float32((timestamps[-1] - timestamps[0]) / raster))
                if int(num) == num:
                    timestamps = linspace(t[0], t[-1], int(num))
                else:
                    timestamps = arange(t[0], t[-1], raster)
        else:
            timestamps = t
        return timestamps

    def get_bus_signal(
        self,
        bus,
        name,
        database=None,
        ignore_invalidation_bits=False,
        data=None,
        raw=False,
        ignore_value2text_conversion=True,
    ):
        """get a signal decoded from a raw bus logging. The currently supported buses are
        CAN and LIN (LDF databases are not supported, they need to be converted to DBC and
        feed to this function)

        .. versionadded:: 6.0.0


        Parameters
        ----------
        bus : str
            "CAN" or "LIN"
        name : str
            signal name
        database : str
            path of external CAN/LIN database file (.dbc or .arxml) or canmatrix.CanMatrix; default *None*

            .. versionchanged:: 6.0.0
                `db` and `database` arguments were merged into this single argument

        ignore_invalidation_bits : bool
            option to ignore invalidation bits
        raw : bool
            return channel samples without applying the conversion rule; default
            `False`
        ignore_value2text_conversion : bool
            return channel samples without values that have a description in .dbc or .arxml file
            `True`

        Returns
        -------
        sig : Signal
            Signal object with the physical values

        """

        if bus == "CAN":
            return self.get_can_signal(
                name,
                database=database,
                ignore_invalidation_bits=ignore_invalidation_bits,
                data=data,
                raw=raw,
                ignore_value2text_conversion=ignore_value2text_conversion,
            )
        elif bus == "LIN":
            return self.get_lin_signal(
                name,
                database=database,
                ignore_invalidation_bits=ignore_invalidation_bits,
                data=data,
                raw=raw,
                ignore_value2text_conversion=ignore_value2text_conversion,
            )

    def get_can_signal(
        self,
        name,
        database=None,
        ignore_invalidation_bits=False,
        data=None,
        raw=False,
        ignore_value2text_conversion=True,
    ):
        """get CAN message signal. You can specify an external CAN database (
        *database* argument) or canmatrix database object that has already been
        loaded from a file (*db* argument).

        The signal name can be specified in the following ways

        * ``CAN<ID>.<MESSAGE_NAME>.<SIGNAL_NAME>`` - the `ID` value starts from 1
          and must match the ID found in the measurement (the source CAN bus ID)
          Example: CAN1.Wheels.FL_WheelSpeed

        * ``CAN<ID>.CAN_DataFrame_<MESSAGE_ID>.<SIGNAL_NAME>`` - the `ID` value
          starts from 1 and the `MESSAGE_ID` is the decimal message ID as found
          in the database. Example: CAN1.CAN_DataFrame_218.FL_WheelSpeed

        * ``<MESSAGE_NAME>.<SIGNAL_NAME>`` - in this case the first occurrence of
          the message name and signal are returned (the same message could be
          found on multiple CAN buses; for example on CAN1 and CAN3)
          Example: Wheels.FL_WheelSpeed

        * ``CAN_DataFrame_<MESSAGE_ID>.<SIGNAL_NAME>`` - in this case the first
          occurrence of the message name and signal are returned (the same
          message could be found on multiple CAN buses; for example on CAN1 and
          CAN3). Example: CAN_DataFrame_218.FL_WheelSpeed

        * ``<SIGNAL_NAME>`` - in this case the first occurrence of the signal
          name is returned (the same signal name could be found in multiple
          messages and on multiple CAN buses). Example: FL_WheelSpeed


        Parameters
        ----------
        name : str
            signal name
        database : str
            path of external CAN database file (.dbc or .arxml) or canmatrix.CanMatrix; default *None*

            .. versionchanged:: 6.0.0
                `db` and `database` arguments were merged into this single argument

        ignore_invalidation_bits : bool
            option to ignore invalidation bits
        raw : bool
            return channel samples without applying the conversion rule; default
            `False`
        ignore_value2text_conversion : bool
            return channel samples without values that have a description in .dbc or .arxml file
            `True`

        Returns
        -------
        sig : Signal
            Signal object with the physical values

        """

        if database is None:
            return self.get(name)

        if isinstance(database, (str, Path)):
            database_path = Path(database)
            if database_path.suffix.lower() not in (".arxml", ".dbc"):
                message = f'Expected .dbc or .arxml file as CAN channel attachment but got "{database_path}"'
                logger.exception(message)
                raise MdfException(message)
            else:
                db_string = database_path.read_bytes()
                md5_sum = md5(db_string).digest()

                if md5_sum in self._external_dbc_cache:
                    db = self._external_dbc_cache[md5_sum]
                else:
                    db = load_can_database(database_path, contents=db_string)
                    if db is None:
                        raise MdfException("failed to load database")
        else:
            db = database

        is_j1939 = db.contains_j1939

        name_ = name.split(".")

        if len(name_) == 3:
            can_id_str, message_id_str, signal = name_

            can_id = v4c.CAN_ID_PATTERN.search(can_id_str)
            if can_id is None:
                raise MdfException(
                    f'CAN id "{can_id_str}" of signal name "{name}" is not recognised by this library'
                )
            else:
                can_id = int(can_id.group("id"))

            message_id = v4c.CAN_DATA_FRAME_PATTERN.search(message_id_str)
            if message_id is None:
                message_id = message_id_str
            else:
                message_id = int(message_id)

            if isinstance(message_id, str):
                message = db.frame_by_name(message_id)
            else:
                message = db.frame_by_id(message_id)

        elif len(name_) == 2:
            message_id_str, signal = name_

            can_id = None

            message_id = v4c.CAN_DATA_FRAME_PATTERN.search(message_id_str)
            if message_id is None:
                message_id = message_id_str
            else:
                message_id = int(message_id.group("id"))

            if isinstance(message_id, str):
                message = db.frame_by_name(message_id)
            else:
                message = db.frame_by_id(message_id)

        else:
            message = None
            for msg in db:
                for signal in msg:
                    if signal.name == name:
                        message = msg

            can_id = None
            signal = name

        if message is None:
            raise MdfException(f"Could not find signal {name} in {database}")

        for sig in message.signals:
            if sig.name == signal:
                signal = sig
                break
        else:
            raise MdfException(
                f'Signal "{signal}" not found in message "{message.name}" of "{database}"'
            )

        if can_id is None:
            index = None
            for _can_id, messages in self.bus_logging_map["CAN"].items():

                if is_j1939:
                    test_ids = [
                        canmatrix.ArbitrationId(id_, extended=True).pgn
                        for id_ in self.bus_logging_map["CAN"][_can_id]
                    ]

                    id_ = message.arbitration_id.pgn

                else:
                    id_ = message.arbitration_id.id
                    test_ids = self.bus_logging_map["CAN"][_can_id]

                if id_ in test_ids:
                    if is_j1939:
                        for id__, idx in self.bus_logging_map["CAN"][_can_id].items():
                            if canmatrix.ArbitrationId(id__, extended=True).pgn == id_:
                                index = idx
                                break
                    else:
                        index = self.bus_logging_map["CAN"][_can_id][
                            message.arbitration_id.id
                        ]

                if index is not None:
                    break
            else:
                raise MdfException(
                    f'Message "{message.name}" (ID={hex(message.arbitration_id.id)}) not found in the measurement'
                )
        else:
            if can_id in self.bus_logging_map["CAN"]:
                if is_j1939:
                    test_ids = [
                        canmatrix.ArbitrationId(id_, extended=True).pgn
                        for id_ in self.bus_logging_map["CAN"][can_id]
                    ]
                    id_ = message.arbitration_id.pgn

                else:
                    id_ = message.arbitration_id.id
                    test_ids = self.bus_logging_map["CAN"][can_id]

                if id_ in test_ids:
                    if is_j1939:
                        for id__, idx in self.bus_logging_map["CAN"][can_id].items():
                            if canmatrix.ArbitrationId(id__, extended=True).pgn == id_:
                                index = idx
                                break
                    else:
                        index = self.bus_logging_map["CAN"][can_id][
                            message.arbitration_id.id
                        ]
                else:
                    raise MdfException(
                        f'Message "{message.name}" (ID={hex(message.arbitration_id.id)}) not found in the measurement'
                    )
            else:
                raise MdfException(
                    f'No logging from "{can_id}" was found in the measurement'
                )

        can_ids = self.get(
            "CAN_DataFrame.ID",
            group=index,
            ignore_invalidation_bits=ignore_invalidation_bits,
            data=data,
        )
        can_ids.samples = can_ids.samples.astype("<u4") & 0x1FFFFFFF

        payload = self.get(
            "CAN_DataFrame.DataBytes",
            group=index,
            samples_only=True,
            ignore_invalidation_bits=ignore_invalidation_bits,
            data=data,
        )[0]

        if is_j1939:
            tmp_pgn = can_ids.samples >> 8
            ps = tmp_pgn & 0xFF
            pf = (can_ids.samples >> 16) & 0xFF
            _pgn = tmp_pgn & 0x3FF00
            can_ids.samples = where(pf >= 240, _pgn + ps, _pgn)

            idx = argwhere(can_ids.samples == message.arbitration_id.pgn).ravel()
        else:
            idx = argwhere(can_ids.samples == message.arbitration_id.id).ravel()

        payload = payload[idx]
        t = can_ids.timestamps[idx].copy()

        if can_ids.invalidation_bits is not None:
            invalidation_bits = can_ids.invalidation_bits[idx]
        else:
            invalidation_bits = None

        if not ignore_invalidation_bits and invalidation_bits is not None:
            payload = payload[nonzero(~invalidation_bits)[0]]
            t = t[nonzero(~invalidation_bits)[0]]

        extracted_signals = extract_mux(
            payload,
            message,
            None,
            None,
            t,
            original_message_id=None,
            ignore_value2text_conversion=ignore_value2text_conversion,
            raw=raw,
        )

        comment = signal.comment or ""

        for entry, signals in extracted_signals.items():
            for name_, sig in signals.items():
                if name_ == signal.name:
                    sig = Signal(
                        samples=sig["samples"],
                        timestamps=sig["t"],
                        name=name,
                        unit=signal.unit or "",
                        comment=comment,
                    )
                    if len(sig):
                        return sig
                    else:
                        raise MdfException(
                            f'No logging from "{signal}" was found in the measurement'
                        )

        raise MdfException(f'No logging from "{signal}" was found in the measurement')

    def get_lin_signal(
        self,
        name,
        database=None,
        ignore_invalidation_bits=False,
        data=None,
        raw=False,
        ignore_value2text_conversion=True,
    ):
        """get LIN message signal. You can specify an external LIN database (
        *database* argument) or canmatrix database object that has already been
        loaded from a file (*db* argument).

        The signal name can be specified in the following ways

        * ``LIN_Frame_<MESSAGE_ID>.<SIGNAL_NAME>`` - Example: LIN_Frame_218.FL_WheelSpeed

        * ``<MESSAGE_NAME>.<SIGNAL_NAME>`` - Example: Wheels.FL_WheelSpeed

        * ``<SIGNAL_NAME>`` - Example: FL_WheelSpeed

        .. versionadded:: 6.0.0


        Parameters
        ----------
        name : str
            signal name
        database : str
            path of external LIN database file (.dbc, .arxml or .ldf) or canmatrix.CanMatrix;
            default *None*

        ignore_invalidation_bits : bool
            option to ignore invalidation bits
        raw : bool
            return channel samples without applying the conversion rule; default
            `False`
        ignore_value2text_conversion : bool
            return channel samples without values that have a description in .dbc, .arxml or .ldf file
            `True`

        Returns
        -------
        sig : Signal
            Signal object with the physical values

        """

        if database is None:
            return self.get(name)

        if isinstance(database, (str, Path)):
            database_path = Path(database)
            if database_path.suffix.lower() not in (".arxml", ".dbc", ".ldf"):
                message = f'Expected .dbc, .arxml or .ldf file as LIN channel attachment but got "{database_path}"'
                logger.exception(message)
                raise MdfException(message)
            else:
                db_string = database_path.read_bytes()
                md5_sum = md5(db_string).digest()

                if md5_sum in self._external_dbc_cache:
                    db = self._external_dbc_cache[md5_sum]
                else:
                    contents = (
                        None if database_path.suffix.lower() == ".ldf" else db_string
                    )
                    db = load_can_database(database_path, contents=contents)
                    if db is None:
                        raise MdfException("failed to load database")
        else:
            db = database

        name_ = name.split(".")

        if len(name_) == 2:
            message_id_str, signal = name_

            message_id = v4c.LIN_DATA_FRAME_PATTERN.search(message_id_str)
            if message_id is None:
                message_id = message_id_str
            else:
                message_id = int(message_id.group("id"))

            if isinstance(message_id, str):
                message = db.frame_by_name(message_id)
            else:
                message = db.frame_by_id(message_id)

        else:
            message = None
            for msg in db:
                for signal in msg:
                    if signal.name == name:
                        message = msg

            signal = name

        if message is None:
            raise MdfException(f"Could not find signal {name} in {database}")

        for sig in message.signals:
            if sig.name == signal:
                signal = sig
                break
        else:
            raise MdfException(
                f'Signal "{signal}" not found in message "{message.name}" of "{database}"'
            )

        id_ = message.arbitration_id.id

        if id_ in self.bus_logging_map["LIN"]:
            index = self.bus_logging_map["LIN"][id_]
        else:
            raise MdfException(
                f'Message "{message.name}" (ID={hex(message.arbitration_id.id)}) not found in the measurement'
            )

        can_ids = self.get(
            "LIN_Frame.ID",
            group=index,
            ignore_invalidation_bits=ignore_invalidation_bits,
            data=data,
        )
        can_ids.samples = can_ids.samples.astype("<u4") & 0x1FFFFFFF
        payload = self.get(
            "LIN_Frame.DataBytes",
            group=index,
            samples_only=True,
            ignore_invalidation_bits=ignore_invalidation_bits,
            data=data,
        )[0]

        idx = argwhere(can_ids.samples == message.arbitration_id.id).ravel()

        payload = payload[idx]
        t = can_ids.timestamps[idx].copy()

        if can_ids.invalidation_bits is not None:
            invalidation_bits = can_ids.invalidation_bits[idx]
        else:
            invalidation_bits = None

        if not ignore_invalidation_bits and invalidation_bits is not None:
            payload = payload[nonzero(~invalidation_bits)[0]]
            t = t[nonzero(~invalidation_bits)[0]]

        extracted_signals = extract_mux(
            payload,
            message,
            None,
            None,
            t,
            original_message_id=None,
            ignore_value2text_conversion=ignore_value2text_conversion,
            raw=raw,
        )

        comment = signal.comment or ""

        for entry, signals in extracted_signals.items():
            for name_, sig in signals.items():
                if name_ == signal.name:
                    sig = Signal(
                        samples=sig["samples"],
                        timestamps=sig["t"],
                        name=name,
                        unit=signal.unit or "",
                        comment=comment,
                    )
                    if len(sig):
                        return sig
                    else:
                        raise MdfException(
                            f'No logging from "{signal}" was found in the measurement'
                        )

        raise MdfException(f'No logging from "{signal}" was found in the measurement')

    def info(self):
        """get MDF information as a dict

        Examples
        --------
        >>> mdf = MDF4('test.mdf')
        >>> mdf.info()


        """
        info = {
            "version": self.version,
            "program": self.identification.program_identification.decode("utf-8").strip(
                " \0\n\r\t"
            ),
            "comment": self.header.comment,
        }
        info["groups"] = len(self.groups)
        for i, gp in enumerate(self.groups):
            inf = {}
            info[f"group {i}"] = inf
            inf["cycles"] = gp.channel_group.cycles_nr
            inf["comment"] = gp.channel_group.comment
            inf["channels count"] = len(gp.channels)
            for j, channel in enumerate(gp.channels):
                name = channel.name

                ch_type = v4c.CHANNEL_TYPE_TO_DESCRIPTION[channel.channel_type]
                inf[f"channel {j}"] = f'name="{name}" type={ch_type}'

        return info

    def save(self, dst, overwrite=False, compression=0):
        """Save MDF to *dst*. If overwrite is *True* then the destination file
        is overwritten, otherwise the file name is appended with '.<cntr>', were
        '<cntr>' is the first counter that produces a new file name
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
        output_file : pathlib.Path
            path to saved file

        """

        if is_file_like(dst):
            dst_ = dst
            file_like = True
            if hasattr(dst, "name"):
                dst = Path(dst.name)
            else:
                dst = Path("__file_like.mf4")
            dst_.seek(0)
            suffix = ".mf4"
        else:
            file_like = False
            suffix = Path(dst).suffix.lower()

            dst = Path(dst).with_suffix(".mf4")

            destination_dir = dst.parent
            destination_dir.mkdir(parents=True, exist_ok=True)

            if overwrite is False:
                if dst.is_file():
                    cntr = 0
                    while True:
                        name = dst.with_suffix(f".{cntr}.mf4")
                        if not name.exists():
                            break
                        else:
                            cntr += 1
                    message = (
                        f'Destination file "{dst}" already exists '
                        f'and "overwrite" is False. Saving MDF file as "{name}"'
                    )
                    logger.warning(message)
                    dst = name

            if dst == self.name:
                destination = dst.with_suffix(".savetemp")
            else:
                destination = dst

            dst_ = open(destination, "wb+")

        if not self.file_history:
            comment = "created"
        else:
            comment = "updated"

        fh = FileHistory()
        fh.comment = f"""<FHcomment>
<TX>{comment}</TX>
<tool_id>asammdf</tool_id>
<tool_vendor>asammdf</tool_vendor>
<tool_version>{__version__}</tool_version>
</FHcomment>"""

        self.file_history.append(fh)

        cg_map = {}

        try:
            defined_texts = {}
            cc_map = {}
            si_map = {}

            groups_nr = len(self.groups)

            write = dst_.write
            tell = dst_.tell
            seek = dst_.seek

            blocks = []

            write(bytes(self.identification))

            self.header.to_blocks(dst_.tell(), blocks)
            for block in blocks:
                write(bytes(block))

            original_data_addresses = []

            if compression == 1:
                zip_type = v4c.FLAG_DZ_DEFLATE
            else:
                zip_type = v4c.FLAG_DZ_TRANPOSED_DEFLATE

            # write DataBlocks first
            for gp_nr, gp in enumerate(self.groups):
                original_data_addresses.append(gp.data_group.data_block_addr)

                if gp.channel_group.flags & v4c.FLAG_CG_VLSD:
                    continue

                address = tell()

                total_size = (
                    gp.channel_group.samples_byte_nr
                    + gp.channel_group.invalidation_bytes_nr
                ) * gp.channel_group.cycles_nr

                if total_size:

                    if self._write_fragment_size:

                        samples_size = (
                            gp.channel_group.samples_byte_nr
                            + gp.channel_group.invalidation_bytes_nr
                        )
                        if samples_size:
                            split_size = self._write_fragment_size // samples_size
                            split_size *= samples_size
                            if split_size == 0:
                                split_size = samples_size
                            chunks = float(total_size) / split_size
                            chunks = int(ceil(chunks))
                        else:
                            chunks = 1
                    else:
                        chunks = 1

                    self.configure(read_fragment_size=split_size)
                    data = self._load_data(gp)

                    if chunks == 1:
                        data_, _1, _2, inval_ = next(data)
                        if self.version >= "4.20" and gp.uses_ld:
                            if compression:
                                if gp.channel_group.samples_byte_nr > 1:
                                    current_zip_type = zip_type
                                    if compression == 1:
                                        param = 0
                                    else:
                                        param = gp.channel_group.samples_byte_nr
                                else:
                                    current_zip_type = v4c.FLAG_DZ_DEFLATE
                                    param = 0

                                kwargs = {
                                    "data": data_,
                                    "zip_type": current_zip_type,
                                    "param": param,
                                    "original_type": b"DV",
                                }
                                data_block = DataZippedBlock(**kwargs)
                            else:
                                data_block = DataBlock(data=data_, type="DV")
                            write(bytes(data_block))
                            data_address = address

                            align = data_block.block_len % 8
                            if align:
                                write(b"\0" * (8 - align))

                            if inval_ is not None:
                                inval_address = address = tell()
                                if compression:
                                    if compression == 1:
                                        param = 0
                                    else:
                                        param = gp.channel_group.invalidation_bytes_nr
                                    kwargs = {
                                        "data": inval_,
                                        "zip_type": zip_type,
                                        "param": param,
                                        "original_type": b"DI",
                                    }
                                    inval_block = DataZippedBlock(**kwargs)
                                else:
                                    inval_block = DataBlock(data=inval_, type="DI")
                                write(bytes(inval_block))

                                align = inval_block.block_len % 8
                                if align:
                                    write(b"\0" * (8 - align))

                            address = tell()

                            kwargs = {
                                "flags": v4c.FLAG_LD_EQUAL_LENGHT,
                                "data_block_nr": 1,
                                "data_block_len": len(data_),
                                "data_block_addr_0": data_address,
                            }
                            if inval_:
                                kwargs["flags"] |= v4c.FLAG_LD_INVALIDATION_PRESENT
                                kwargs["invalidation_bits_addr_0"] = inval_address
                            ld_block = ListData(**kwargs)
                            write(bytes(ld_block))

                            align = ld_block.block_len % 8
                            if align:
                                write(b"\0" * (8 - align))

                            if gp.channel_group.cycles_nr:
                                gp.data_group.data_block_addr = address
                            else:
                                gp.data_group.data_block_addr = 0

                        else:
                            if compression and self.version >= "4.10":
                                if compression == 1:
                                    param = 0
                                else:
                                    param = (
                                        gp.channel_group.samples_byte_nr
                                        + gp.channel_group.invalidation_bytes_nr
                                    )
                                kwargs = {
                                    "data": data_,
                                    "zip_type": zip_type,
                                    "param": param,
                                }
                                data_block = DataZippedBlock(**kwargs)
                            else:
                                data_block = DataBlock(data=data_)
                            write(bytes(data_block))

                            align = data_block.block_len % 8
                            if align:
                                write(b"\0" * (8 - align))

                            if gp.channel_group.cycles_nr:
                                gp.data_group.data_block_addr = address
                            else:
                                gp.data_group.data_block_addr = 0
                    else:
                        if self.version >= "4.20" and gp.uses_ld:
                            dv_addr = []
                            di_addr = []
                            block_size = 0
                            for i, (data_, _1, _2, inval_) in enumerate(data):
                                if i == 0:
                                    block_size = len(data_)
                                if compression:
                                    if compression == 1:
                                        param = 0
                                    else:
                                        param = gp.channel_group.samples_byte_nr
                                    kwargs = {
                                        "data": data_,
                                        "zip_type": zip_type,
                                        "param": param,
                                        "original_type": b"DV",
                                    }
                                    data_block = DataZippedBlock(**kwargs)
                                else:
                                    data_block = DataBlock(data=data_, type="DV")
                                dv_addr.append(tell())
                                write(bytes(data_block))

                                align = data_block.block_len % 8
                                if align:
                                    write(b"\0" * (8 - align))

                                if inval_ is not None:

                                    if compression:
                                        if compression == 1:
                                            param = 0
                                        else:
                                            param = (
                                                gp.channel_group.invalidation_bytes_nr
                                            )
                                        kwargs = {
                                            "data": inval_,
                                            "zip_type": zip_type,
                                            "param": param,
                                            "original_type": b"DI",
                                        }
                                        inval_block = DataZippedBlock(**kwargs)
                                    else:
                                        inval_block = DataBlock(data=inval_, type="DI")
                                    di_addr.append(tell())
                                    write(bytes(inval_block))

                                    align = inval_block.block_len % 8
                                    if align:
                                        write(b"\0" * (8 - align))

                            address = tell()

                            kwargs = {
                                "flags": v4c.FLAG_LD_EQUAL_LENGHT,
                                "data_block_nr": len(dv_addr),
                                "data_block_len": block_size,
                            }
                            for i, addr in enumerate(dv_addr):
                                kwargs[f"data_block_addr_{i}"] = addr

                            if di_addr:
                                kwargs["flags"] |= v4c.FLAG_LD_INVALIDATION_PRESENT
                                for i, addr in enumerate(di_addr):
                                    kwargs[f"invalidation_bits_addr_{i}"] = addr

                            ld_block = ListData(**kwargs)
                            write(bytes(ld_block))

                            align = ld_block.block_len % 8
                            if align:
                                write(b"\0" * (8 - align))

                            if gp.channel_group.cycles_nr:
                                gp.data_group.data_block_addr = address
                            else:
                                gp.data_group.data_block_addr = 0

                        else:
                            kwargs = {
                                "flags": v4c.FLAG_DL_EQUAL_LENGHT,
                                "zip_type": zip_type,
                            }
                            hl_block = HeaderList(**kwargs)

                            kwargs = {
                                "flags": v4c.FLAG_DL_EQUAL_LENGHT,
                                "links_nr": chunks + 1,
                                "data_block_nr": chunks,
                                "data_block_len": split_size,
                            }
                            dl_block = DataList(**kwargs)

                            for i, data__ in enumerate(data):
                                data_ = data__[0]

                                if compression and self.version >= "4.10":
                                    if compression == 1:
                                        zip_type = v4c.FLAG_DZ_DEFLATE
                                    else:
                                        zip_type = v4c.FLAG_DZ_TRANPOSED_DEFLATE
                                    if compression == 1:
                                        param = 0
                                    else:
                                        param = (
                                            gp.channel_group.samples_byte_nr
                                            + gp.channel_group.invalidation_bytes_nr
                                        )
                                    kwargs = {
                                        "data": data_,
                                        "zip_type": zip_type,
                                        "param": param,
                                    }
                                    block = DataZippedBlock(**kwargs)
                                else:
                                    block = DataBlock(data=data_)
                                address = tell()
                                block.address = address

                                write(bytes(block))

                                align = block.block_len % 8
                                if align:
                                    write(b"\0" * (8 - align))
                                dl_block[f"data_block_addr{i}"] = address

                            address = tell()
                            dl_block.address = address
                            write(bytes(dl_block))

                            if compression and self.version != "4.00":
                                hl_block.first_dl_addr = address
                                address = tell()
                                hl_block.address = address
                                write(bytes(hl_block))

                            gp.data_group.data_block_addr = address
                else:
                    gp.data_group.data_block_addr = 0

                if self._callback:
                    self._callback(int(50 * (gp_nr + 1) / groups_nr), 100)
                if self._terminate:
                    dst_.close()
                    self.close()
                    return

            address = tell()

            blocks = []

            # file history blocks
            for fh in self.file_history:
                address = fh.to_blocks(address, blocks, defined_texts)

            for i, fh in enumerate(self.file_history[:-1]):
                fh.next_fh_addr = self.file_history[i + 1].address
            self.file_history[-1].next_fh_addr = 0

            # data groups
            gp_rec_ids = []
            valid_data_groups = []
            for gp in self.groups:
                if gp.channel_group.flags & v4c.FLAG_CG_VLSD:
                    continue

                valid_data_groups.append(gp.data_group)
                gp_rec_ids.append(gp.data_group.record_id_len)

                address = gp.data_group.to_blocks(address, blocks, defined_texts)

            if valid_data_groups:
                for i, dg in enumerate(valid_data_groups[:-1]):
                    addr_ = valid_data_groups[i + 1].address
                    dg.next_dg_addr = addr_
                valid_data_groups[-1].next_dg_addr = 0

            # go through each data group and append the rest of the blocks
            for i, gp in enumerate(self.groups):

                channels = gp.channels

                for j, channel in enumerate(channels):
                    if channel.attachment is not None:
                        channel.attachment_addr = self.attachments[
                            channel.attachment
                        ].address
                    elif channel.attachment_nr:
                        channel.attachment_addr = 0

                    address = channel.to_blocks(
                        address, blocks, defined_texts, cc_map, si_map
                    )

                    if channel.channel_type == v4c.CHANNEL_TYPE_SYNC:
                        if channel.attachment is not None:
                            channel.data_block_addr = self.attachments[
                                channel.attachment
                            ].address
                    else:

                        sdata = self._load_signal_data(group=gp, index=j)
                        if sdata:
                            split_size = self._write_fragment_size
                            if self._write_fragment_size:
                                chunks = float(len(sdata)) / split_size
                                chunks = int(ceil(chunks))
                            else:
                                chunks = 1

                            if chunks == 1:
                                if compression and self.version > "4.00":
                                    signal_data = DataZippedBlock(
                                        data=sdata,
                                        zip_type=v4c.FLAG_DZ_DEFLATE,
                                        original_type=b"SD",
                                    )
                                    signal_data.address = address
                                    address += signal_data.block_len
                                    blocks.append(signal_data)
                                    align = signal_data.block_len % 8
                                    if align:
                                        blocks.append(b"\0" * (8 - align))
                                        address += 8 - align
                                else:
                                    signal_data = DataBlock(data=sdata, type="SD")
                                    signal_data.address = address
                                    address += signal_data.block_len
                                    blocks.append(signal_data)
                                    align = signal_data.block_len % 8
                                    if align:
                                        blocks.append(b"\0" * (8 - align))
                                        address += 8 - align

                                channel.data_block_addr = signal_data.address
                            else:
                                kwargs = {
                                    "flags": v4c.FLAG_DL_EQUAL_LENGHT,
                                    "links_nr": chunks + 1,
                                    "data_block_nr": chunks,
                                    "data_block_len": self._write_fragment_size,
                                }
                                dl_block = DataList(**kwargs)

                                for k in range(chunks):

                                    data_ = sdata[k * split_size : (k + 1) * split_size]
                                    if compression and self.version > "4.00":
                                        zip_type = v4c.FLAG_DZ_DEFLATE
                                        param = 0

                                        kwargs = {
                                            "data": data_,
                                            "zip_type": zip_type,
                                            "param": param,
                                            "original_type": b"SD",
                                        }
                                        block = DataZippedBlock(**kwargs)
                                    else:
                                        block = DataBlock(data=data_, type="SD")
                                    blocks.append(block)
                                    block.address = address
                                    address += block.block_len

                                    align = block.block_len % 8
                                    if align:
                                        blocks.append(b"\0" * (8 - align))
                                        address += 8 - align
                                    dl_block[f"data_block_addr{k}"] = block.address

                                dl_block.address = address
                                blocks.append(dl_block)

                                address += dl_block.block_len

                                if compression and self.version > "4.00":
                                    kwargs = {
                                        "flags": v4c.FLAG_DL_EQUAL_LENGHT,
                                        "zip_type": v4c.FLAG_DZ_DEFLATE,
                                        "first_dl_addr": dl_block.address,
                                    }
                                    hl_block = HeaderList(**kwargs)
                                    hl_block.address = address
                                    address += hl_block.block_len

                                    blocks.append(hl_block)

                                    channel.data_block_addr = hl_block.address
                                else:
                                    channel.data_block_addr = dl_block.address

                        else:
                            channel.data_block_addr = 0

                    dep_list = gp.channel_dependencies[j]
                    if dep_list:
                        if all(isinstance(dep, ChannelArrayBlock) for dep in dep_list):
                            for dep in dep_list:
                                dep.address = address
                                address += dep.block_len
                                blocks.append(dep)
                            for k, dep in enumerate(dep_list[:-1]):
                                dep.composition_addr = dep_list[k + 1].address
                            dep_list[-1].composition_addr = 0

                            channel.component_addr = dep_list[0].address

                        else:
                            index = dep_list[0][1]
                            addr_ = gp.channels[index].address

                group_channels = gp.channels
                if group_channels:
                    for j, channel in enumerate(group_channels[:-1]):
                        channel.next_ch_addr = group_channels[j + 1].address
                    group_channels[-1].next_ch_addr = 0

                # channel dependecies
                j = len(channels) - 1
                while j >= 0:
                    dep_list = gp.channel_dependencies[j]
                    if dep_list and all(isinstance(dep, tuple) for dep in dep_list):
                        index = dep_list[0][1]
                        channels[j].component_addr = channels[index].address
                        index = dep_list[-1][1]
                        channels[j].next_ch_addr = channels[index].next_ch_addr
                        channels[index].next_ch_addr = 0

                        for _, ch_nr in dep_list:
                            channels[ch_nr].source_addr = 0
                    j -= 1

                # channel group
                if gp.channel_group.flags & v4c.FLAG_CG_VLSD:
                    continue

                gp.channel_group.first_sample_reduction_addr = 0

                if channels:
                    gp.channel_group.first_ch_addr = gp.channels[0].address
                else:
                    gp.channel_group.first_ch_addr = 0
                gp.channel_group.next_cg_addr = 0

                address = gp.channel_group.to_blocks(
                    address, blocks, defined_texts, si_map
                )
                gp.data_group.first_cg_addr = gp.channel_group.address

                cg_map[i] = gp.channel_group.address

                if self._callback:
                    self._callback(int(50 * (i + 1) / groups_nr) + 25, 100)
                if self._terminate:
                    dst_.close()
                    self.close()
                    return

            for gp in self.groups:
                for dep_list in gp.channel_dependencies:
                    if dep_list:
                        if all(isinstance(dep, ChannelArrayBlock) for dep in dep_list):
                            for dep in dep_list:

                                for i, (gp_nr, ch_nr) in enumerate(
                                    dep.dynamic_size_channels
                                ):
                                    grp = self.groups[gp_nr]
                                    ch = grp.channels[ch_nr]
                                    dep[
                                        f"dynamic_size_{i}_dg_addr"
                                    ] = grp.data_group.address
                                    dep[
                                        f"dynamic_size_{i}_cg_addr"
                                    ] = grp.channel_group.address
                                    dep[f"dynamic_size_{i}_ch_addr"] = ch.address

                                for i, (gp_nr, ch_nr) in enumerate(
                                    dep.input_quantity_channels
                                ):
                                    grp = self.groups[gp_nr]
                                    ch = grp.channels[ch_nr]
                                    dep[
                                        f"input_quantity_{i}_dg_addr"
                                    ] = grp.data_group.address
                                    dep[
                                        f"input_quantity_{i}_cg_addr"
                                    ] = grp.channel_group.address
                                    dep[f"input_quantity_{i}_ch_addr"] = ch.address

                                for i, conversion in enumerate(dep.axis_conversions):
                                    if conversion:
                                        address = conversion.to_blocks(
                                            address, blocks, defined_texts, cc_map
                                        )
                                        dep[f"axis_conversion_{i}"] = conversion.address
                                    else:
                                        dep[f"axis_conversion_{i}"] = 0

                                if dep.output_quantity_channel:
                                    gp_nr, ch_nr = dep.output_quantity_channel
                                    grp = self.groups[gp_nr]
                                    ch = grp.channels[ch_nr]
                                    dep[
                                        f"output_quantity_dg_addr"
                                    ] = grp.data_group.address
                                    dep[
                                        f"output_quantity_cg_addr"
                                    ] = grp.channel_group.address
                                    dep[f"output_quantity_ch_addr"] = ch.address

                                if dep.comparison_quantity_channel:
                                    gp_nr, ch_nr = dep.comparison_quantity_channel
                                    grp = self.groups[gp_nr]
                                    ch = grp.channels[ch_nr]
                                    dep[
                                        f"comparison_quantity_dg_addr"
                                    ] = grp.data_group.address
                                    dep[
                                        f"comparison_quantity_cg_addr"
                                    ] = grp.channel_group.address
                                    dep[f"comparison_quantity_ch_addr"] = ch.address

                                for i, (gp_nr, ch_nr) in enumerate(dep.axis_channels):
                                    grp = self.groups[gp_nr]
                                    ch = grp.channels[ch_nr]
                                    dep[
                                        f"scale_axis_{i}_dg_addr"
                                    ] = grp.data_group.address
                                    dep[
                                        f"scale_axis_{i}_cg_addr"
                                    ] = grp.channel_group.address
                                    dep[f"scale_axis_{i}_ch_addr"] = ch.address

            position = tell()

            for gp in self.groups:
                gp.data_group.record_id_len = 0

                cg_master_index = gp.channel_group.cg_master_index
                if cg_master_index is not None:
                    gp.channel_group.cg_master_addr = cg_map[cg_master_index]
                    seek(gp.channel_group.address)
                    write(bytes(gp.channel_group))

            seek(position)

            ev_map = []

            if self.events:
                for event in self.events:
                    for i, ref in enumerate(event.scopes):
                        try:
                            dg_cntr, ch_cntr = ref
                            event[f"scope_{i}_addr"] = (
                                self.groups[dg_cntr].channels[ch_cntr].address
                            )
                        except TypeError:
                            dg_cntr = ref
                            event[f"scope_{i}_addr"] = self.groups[
                                dg_cntr
                            ].channel_group.address

                    blocks.append(event)
                    ev_map.append(address)
                    event.address = address
                    address += event.block_len

                    if event.name:
                        tx_block = TextBlock(text=event.name)
                        tx_block.address = address
                        blocks.append(tx_block)
                        address += tx_block.block_len
                        event.name_addr = tx_block.address
                    else:
                        event.name_addr = 0

                    if event.comment:
                        meta = event.comment.startswith("<EVcomment")
                        tx_block = TextBlock(text=event.comment, meta=meta)
                        tx_block.address = address
                        blocks.append(tx_block)
                        address += tx_block.block_len
                        event.comment_addr = tx_block.address
                    else:
                        event.comment_addr = 0

                    if event.parent is not None:
                        event.parent_ev_addr = ev_map[event.parent]
                    if event.range_start is not None:
                        event.range_start_ev_addr = ev_map[event.range_start]

                for i in range(len(self.events) - 1):
                    self.events[i].next_ev_addr = self.events[i + 1].address
                self.events[-1].next_ev_addr = 0

                self.header.first_event_addr = self.events[0].address

            if self._terminate:
                dst_.close()
                self.close()
                return

            # attachments
            at_map = {}
            if self.attachments:

                # put the attachment texts before the attachments
                for at_block in self.attachments:
                    for text in (at_block.file_name, at_block.mime, at_block.comment):
                        if text not in defined_texts:
                            tx_block = TextBlock(text=str(text))
                            defined_texts[text] = address
                            tx_block.address = address
                            address += tx_block.block_len
                            blocks.append(tx_block)

                for at_block in self.attachments:
                    address = at_block.to_blocks(address, blocks, defined_texts)

                for i in range(len(self.attachments) - 1):
                    at_block = self.attachments[i]
                    at_block.next_at_addr = self.attachments[i + 1].address
                self.attachments[-1].next_at_addr = 0

                if self.events:
                    for event in self.events:
                        for i in range(event.attachment_nr):
                            key = f"attachment_{i}_addr"
                            addr = event[key]
                            event[key] = at_map[addr]

                for i, gp in enumerate(self.groups):

                    for j, channel in enumerate(gp.channels):
                        if channel.attachment is not None:
                            channel.attachment_addr = self.attachments[
                                channel.attachment
                            ].address
                        elif channel.attachment_nr:
                            channel.attachment_addr = 0

                        if (
                            channel.channel_type == v4c.CHANNEL_TYPE_SYNC
                            and channel.attachment is not None
                        ):
                            channel.data_block_addr = self.attachments[
                                channel.attachment
                            ].address

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
                gp.data_group.record_id_len = rec_id

            if valid_data_groups:
                addr_ = valid_data_groups[0].address
                self.header.first_dg_addr = addr_
            else:
                self.header.first_dg_addr = 0
            self.header.file_history_addr = self.file_history[0].address
            if self.attachments:
                first_attachment = self.attachments[0]
                addr_ = first_attachment.address
                self.header.first_attachment_addr = addr_
            else:
                self.header.first_attachment_addr = 0

            seek(v4c.IDENTIFICATION_BLOCK_SIZE)
            write(bytes(self.header))

            for orig_addr, gp in zip(original_data_addresses, self.groups):
                gp.data_group.data_block_addr = orig_addr

            at_map = {value: key for key, value in at_map.items()}

            for event in self.events:
                for i in range(event.attachment_nr):
                    key = f"attachment_{i}_addr"
                    addr = event[key]
                    event[key] = at_map[addr]

        except:
            if not file_like:
                dst_.close()
            raise
        else:
            if not file_like:
                dst_.close()

        if suffix in (".zip", ".mf4z"):
            output_fname = dst.with_suffix(suffix)
            try:
                zipped_mf4 = ZipFile(output_fname, "w", compression=ZIP_DEFLATED)
                zipped_mf4.write(
                    str(dst),
                    dst.name,
                    compresslevel=1,
                )
                zipped_mf4.close()
                os.remove(destination)
                dst = output_fname
            except:
                pass

        if dst == self.name:
            self.close()
            try:
                Path.unlink(self.name)
                Path.rename(destination, self.name)
            except:
                pass

            self.groups.clear()
            self.header = None
            self.identification = None
            self.file_history.clear()
            self.channels_db.clear()
            self.masters_db.clear()
            self.attachments.clear()
            self.file_comment = None

            self._ch_map.clear()

            self._tempfile = TemporaryFile()
            self._file = open(self.name, "rb")
            self._read()

        if self._callback:
            self._callback(100, 100)

        if self.name == Path("__new__.mf4"):
            self.name = dst

        return dst

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
        gp_nr, ch_nr = self._validate_channel_selection(None, group, index)

        return self.groups[gp_nr].channels[ch_nr].name

    def get_channel_metadata(self, name=None, group=None, index=None):
        gp_nr, ch_nr = self._validate_channel_selection(name, group, index)

        grp = self.groups[gp_nr]

        channel = grp.channels[ch_nr]

        return channel

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
        gp_nr, ch_nr = self._validate_channel_selection(name, group, index)

        grp = self.groups[gp_nr]

        channel = grp.channels[ch_nr]

        conversion = channel.conversion

        unit = conversion and conversion.unit or channel.unit or ""

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
        gp_nr, ch_nr = self._validate_channel_selection(name, group, index)

        grp = self.groups[gp_nr]

        channel = grp.channels[ch_nr]

        return extract_cncomment_xml(channel.comment)

    def _finalize(self):
        """
        Attempt finalization of the file.
        :return:    None
        """

        flags = self.identification.unfinalized_standard_flags

        stream = self._file
        blocks, block_groups, addresses = all_blocks_addresses(stream)

        stream.seek(0, 2)
        limit = stream.tell()
        mapped = self._mapped

        if flags & v4c.FLAG_UNFIN_UPDATE_LAST_DL:
            for dg_addr in block_groups[b"##DG"]:
                group = DataGroup(address=dg_addr, stream=stream, mapped=mapped)
                data_addr = group.data_block_addr
                if not data_addr:
                    continue

                stream.seek(data_addr)
                blk_id = stream.read(4)
                if blk_id == b"##DT":
                    continue
                elif blk_id in (b"##DL", b"##HL"):

                    if blk_id == b"##HL":
                        hl = HeaderList(address=data_addr, stream=stream, mapped=mapped)
                        data_addr = hl.first_dl_addr

                    while True:
                        dl = DataList(address=data_addr, stream=stream, mapped=mapped)
                        if not dl.next_dl_addr:
                            break

                    kwargs = {}

                    count = dl.links_nr - 1
                    valid_count = 0
                    for i in range(count):
                        dt_addr = dl[f"data_block_addr{i}"]
                        if dt_addr:
                            valid_count += 1
                            kwargs[f"data_block_addr{i}"] = dt_addr
                        else:
                            break

                    starting_address = dl.address
                    next_block_position = bisect.bisect_right(
                        addresses, starting_address
                    )
                    # search for data blocks after the DLBLOCK
                    for j in range(i, count):

                        if next_block_position >= len(addresses):
                            break

                        next_block_address = addresses[next_block_position]
                        next_block_type = blocks[next_block_address]

                        if next_block_type not in {b"##DZ", b"##DT", b"##DV", b"##DI"}:
                            break
                        else:

                            stream.seek(next_block_address)

                            if next_block_type == b"##DZ":
                                (
                                    original_type,
                                    zip_type,
                                    param,
                                    original_size,
                                    zip_size,
                                ) = v4c.DZ_COMMON_INFO_uf(
                                    stream.read(v4c.DZ_COMMON_SIZE)
                                )

                                exceeded = (
                                    limit
                                    - (
                                        next_block_address
                                        + v4c.DZ_COMMON_SIZE
                                        + zip_size
                                    )
                                    < 0
                                )

                            else:
                                id_string, block_len = COMMON_SHORT_uf(
                                    stream.read(v4c.COMMON_SIZE)
                                )
                                original_size = block_len - 24

                                exceeded = limit - (next_block_address + block_len) < 0

                            # update the data block size in case all links were NULL before
                            if i == 0 and (dl.flags & v4c.FLAG_DL_EQUAL_LENGHT):
                                kwargs["data_block_len"] = original_size

                            # check if the file limit is exceeded
                            if exceeded:
                                break
                            else:
                                next_block_position += 1
                                valid_count += 1
                                kwargs[f"data_block_addr{j}"] = next_block_address

                    kwargs["links_nr"] = valid_count + 1
                    kwargs["flags"] = dl.flags
                    if dl.flags & v4c.FLAG_DL_EQUAL_LENGHT:
                        kwargs["data_block_len"] = dl.data_block_len
                    else:
                        for i in enumerate(valid_count):
                            kwargs[f"offset_{i}"] = dl[f"offset_{i}"]

                    stream.seek(data_addr)
                    stream.write(bytes(DataList(**kwargs)))

            self.identification[
                "unfinalized_standard_flags"
            ] -= v4c.FLAG_UNFIN_UPDATE_LAST_DL

        if flags & v4c.FLAG_UNFIN_UPDATE_LAST_DT_LENGTH:
            try:
                for dg_addr in block_groups[b"##DG"]:
                    group = DataGroup(address=dg_addr, stream=stream, mapped=mapped)
                    data_addr = group.data_block_addr
                    if not data_addr:
                        continue

                    stream.seek(data_addr)
                    blk_id = stream.read(4)
                    if blk_id == b"##DT":
                        blk = DataBlock(address=data_addr, stream=stream, mapped=mapped)
                    elif blk_id == b"##DL":
                        while True:
                            dl = DataList(
                                address=data_addr, stream=stream, mapped=mapped
                            )
                            if not dl.next_dl_addr:
                                break

                        data_addr = dl[f"data_block_addr{dl.links_nr - 2}"]
                        blk = DataBlock(address=data_addr, stream=stream, mapped=mapped)

                    elif blk_id == b"##HL":

                        hl = HeaderList(address=data_addr, stream=stream, mapped=mapped)

                        data_addr = hl.first_dl_addr
                        while True:
                            dl = DataList(
                                address=data_addr, stream=stream, mapped=mapped
                            )
                            if not dl.next_dl_addr:
                                break

                        data_addr = dl[f"data_block_addr{dl.links_nr - 2}"]
                        blk = DataBlock(address=data_addr, stream=stream, mapped=mapped)

                    next_block = bisect.bisect_right(addresses, data_addr)
                    if next_block == len(addresses):
                        block_len = limit - data_addr
                    else:
                        block_len = addresses[next_block] - data_addr

                    blk.block_len = block_len
                    stream.seek(data_addr)
                    stream.write(bytes(blk))
            except:
                print(format_exc())
                raise

            self.identification.unfinalized_standard_flags -= (
                v4c.FLAG_UNFIN_UPDATE_LAST_DT_LENGTH
            )
        self.identification.file_identification = b"MDF     "

    def _sort(self):
        if self._file is None:
            return

        flags = self.identification["unfinalized_standard_flags"]

        common = defaultdict(list)
        for i, group in enumerate(self.groups):
            if group.sorted:
                continue

            if group.data_blocks:
                address = group.data_blocks[0].address

                common[address].append((i, group.channel_group.record_id))

        read = self._file.read
        seek = self._file.seek

        self._tempfile.seek(0, 2)

        tell = self._tempfile.tell
        write = self._tempfile.write

        for address, groups in common.items():

            cg_map = {
                rec_id: self.groups[index_].channel_group for index_, rec_id in groups
            }

            final_records = {id_: [] for (_, id_) in groups}

            for rec_id, channel_group in cg_map.items():
                if channel_group.address in self._cn_data_map:
                    dg_cntr, ch_cntr = self._cn_data_map[channel_group.address]
                    self.groups[dg_cntr].signal_data[ch_cntr] = []

            group = self.groups[groups[0][0]]

            record_id_nr = group.data_group.record_id_len
            cg_size = group.record_size

            if record_id_nr == 1:
                _unpack_stuct = UINT8_uf
            elif record_id_nr == 2:
                _unpack_stuct = UINT16_uf
            elif record_id_nr == 4:
                _unpack_stuct = UINT32_uf
            elif record_id_nr == 8:
                _unpack_stuct = UINT64_uf
            else:
                message = f"invalid record id size {record_id_nr}"
                raise MdfException(message)

            rem = b""
            for info in group.data_blocks:
                dtblock_address, dtblock_raw_size, dtblock_size, block_type, param = (
                    info.address,
                    info.original_size,
                    info.compressed_size,
                    info.block_type,
                    info.param,
                )

                seek(dtblock_address)

                if block_type != v4c.DT_BLOCK:
                    partial_records = {id_: [] for _, id_ in groups}
                    new_data = read(dtblock_size)

                    if block_type == v4c.DZ_BLOCK_DEFLATE:
                        new_data = decompress(new_data, 0, dtblock_raw_size)
                    elif block_type == v4c.DZ_BLOCK_TRANSPOSED:
                        new_data = decompress(new_data, 0, dtblock_raw_size)
                        cols = param
                        lines = dtblock_raw_size // cols

                        nd = fromstring(new_data[: lines * cols], dtype=uint8)
                        nd = nd.reshape((cols, lines))
                        new_data = nd.T.tobytes() + new_data[lines * cols :]

                    new_data = rem + new_data

                    try:
                        rem = sort_data_block(
                            new_data,
                            partial_records,
                            cg_size,
                            record_id_nr,
                            _unpack_stuct,
                        )
                    except:
                        print(format_exc())
                        raise

                    for rec_id, new_data in partial_records.items():

                        channel_group = cg_map[rec_id]

                        if channel_group.address in self._cn_data_map:
                            dg_cntr, ch_cntr = self._cn_data_map[channel_group.address]
                        else:
                            dg_cntr, ch_cntr = None, None

                        if new_data:

                            tempfile_address = tell()

                            new_data = b"".join(new_data)
                            original_size = len(new_data)
                            if original_size:
                                new_data = lz_compress(new_data)
                                compressed_size = len(new_data)

                                write(new_data)

                                if dg_cntr is not None:

                                    info = SignalDataBlockInfo(
                                        address=tempfile_address,
                                        compressed_size=compressed_size,
                                        original_size=original_size,
                                        block_type=v4c.DZ_BLOCK_LZ,
                                        location=v4c.LOCATION_TEMPORARY_FILE,
                                    )
                                    self.groups[dg_cntr].signal_data[ch_cntr].append(
                                        info
                                    )

                                else:

                                    block_info = DataBlockInfo(
                                        address=tempfile_address,
                                        block_type=v4c.DZ_BLOCK_LZ,
                                        compressed_size=compressed_size,
                                        original_size=original_size,
                                        param=0,
                                    )
                                    final_records[rec_id].append(block_info)

                else:  # DTBLOCK

                    seek(dtblock_address)
                    limit = 32 * 1024 * 1024  # 32MB
                    while dtblock_size:

                        if dtblock_size > limit:
                            dtblock_size -= limit
                            new_data = rem + read(limit)
                        else:
                            new_data = rem + read(dtblock_size)
                            dtblock_size = 0
                        partial_records = {id_: [] for _, id_ in groups}

                        rem = sort_data_block(
                            new_data,
                            partial_records,
                            cg_size,
                            record_id_nr,
                            _unpack_stuct,
                        )

                        for rec_id, new_data in partial_records.items():

                            channel_group = cg_map[rec_id]

                            if channel_group.address in self._cn_data_map:
                                dg_cntr, ch_cntr = self._cn_data_map[
                                    channel_group.address
                                ]
                            else:
                                dg_cntr, ch_cntr = None, None

                            if new_data:

                                tempfile_address = tell()
                                new_data = b"".join(new_data)

                                original_size = len(new_data)
                                if original_size:
                                    new_data = lz_compress(new_data)
                                    compressed_size = len(new_data)

                                    write(new_data)

                                    if dg_cntr is not None:

                                        info = SignalDataBlockInfo(
                                            address=tempfile_address,
                                            compressed_size=compressed_size,
                                            original_size=original_size,
                                            block_type=v4c.DZ_BLOCK_LZ,
                                            location=v4c.LOCATION_TEMPORARY_FILE,
                                        )
                                        self.groups[dg_cntr].signal_data[
                                            ch_cntr
                                        ].append(info)

                                    else:
                                        block_info = DataBlockInfo(
                                            address=tempfile_address,
                                            block_type=v4c.DZ_BLOCK_LZ,
                                            compressed_size=compressed_size,
                                            original_size=original_size,
                                            param=None,
                                        )

                                        final_records[rec_id].append(block_info)

            # after we read all DTBLOCKs in the original file,
            # we assign freshly created blocks from temporary file to
            # corresponding groups.
            for idx, rec_id in groups:
                group = self.groups[idx]
                group.data_location = v4c.LOCATION_TEMPORARY_FILE
                group.set_blocks_info(final_records[rec_id])
                group.sorted = True

        for i, group in enumerate(self.groups):
            if flags & v4c.FLAG_UNFIN_UPDATE_CG_COUNTER:

                channel_group = group.channel_group

                if channel_group.flags & v4c.FLAG_CG_VLSD:
                    continue

                if (
                    self.version >= "4.20"
                    and channel_group.flags & v4c.FLAG_CG_REMOTE_MASTER
                ):
                    index = channel_group.cg_master_index
                else:
                    index = i

                if group.uses_ld:
                    samples_size = channel_group.samples_byte_nr
                else:
                    samples_size = (
                        channel_group.samples_byte_nr
                        + channel_group.invalidation_bytes_nr
                    )

                total_size = sum(blk.original_size for blk in group.data_blocks)

                cycles_nr = total_size // samples_size
                virtual_channel_group = self.virtual_groups[index]
                virtual_channel_group.cycles_nr = cycles_nr
                channel_group.cycles_nr = cycles_nr

        if (
            self.identification["unfinalized_standard_flags"]
            & v4c.FLAG_UNFIN_UPDATE_CG_COUNTER
        ):
            self.identification[
                "unfinalized_standard_flags"
            ] -= v4c.FLAG_UNFIN_UPDATE_CG_COUNTER
        if (
            self.identification["unfinalized_standard_flags"]
            & v4c.FLAG_UNFIN_UPDATE_VLSD_BYTES
        ):
            self.identification[
                "unfinalized_standard_flags"
            ] -= v4c.FLAG_UNFIN_UPDATE_VLSD_BYTES

    def _process_bus_logging(self):
        groups_count = len(self.groups)
        for index in range(groups_count):
            group = self.groups[index]
            if group.channel_group.flags & v4c.FLAG_CG_BUS_EVENT:
                source = group.channel_group.acq_source

                if (
                    source
                    and source.bus_type in (v4c.BUS_TYPE_CAN, v4c.BUS_TYPE_OTHER)
                    and "CAN_DataFrame" in [ch.name for ch in group.channels]
                ):
                    self._process_can_logging(index, group)

                if (
                    source
                    and source.bus_type in (v4c.BUS_TYPE_LIN, v4c.BUS_TYPE_OTHER)
                    and "LIN_Frame" in [ch.name for ch in group.channels]
                ):
                    self._process_lin_logging(index, group)

    def _process_can_logging(self, group_index, grp):

        channels = grp.channels
        group = grp

        dbc = None

        for i, channel in enumerate(channels):
            if channel.name == "CAN_DataFrame":
                attachment_addr = channel.attachment

                if attachment_addr is not None:
                    if attachment_addr not in self._dbc_cache:

                        attachment, at_name, md5_sum = self.extract_attachment(
                            index=attachment_addr,
                            decryption_function=self._decryption_function,
                        )
                        if at_name.suffix.lower() not in (".arxml", ".dbc"):
                            message = f'Expected .dbc or .arxml file as CAN channel attachment but got "{at_name}"'
                            logger.warning(message)
                        elif not attachment:
                            message = f'Attachment "{at_name}" not found'
                            logger.warning(message)
                        else:
                            dbc = load_can_database(at_name, contents=attachment)
                            if dbc:
                                self._dbc_cache[attachment_addr] = dbc
                    else:
                        dbc = self._dbc_cache[attachment_addr]
                break

        if dbc is None:
            parents, dtypes = self._prepare_record(group)
            data = self._load_data(group, optimize_read=False)

            for fragment_index, fragment in enumerate(data):
                if dtypes.itemsize:
                    group.record = fromstring(fragment[0], dtype=dtypes)
                else:
                    group.record = None

                    return

                self._set_temporary_master(None)
                self._set_temporary_master(self.get_master(group_index, data=fragment))

                bus_ids = self.get(
                    "CAN_DataFrame.BusChannel",
                    group=group_index,
                    data=fragment,
                    samples_only=True,
                )[0].astype("<u1")

                msg_ids = (
                    self.get(
                        "CAN_DataFrame.ID",
                        group=group_index,
                        data=fragment,
                        samples_only=True,
                    )[0].astype("<u4")
                    & 0x1FFFFFFF
                )

                if len(bus_ids) == 0:
                    continue

                buses = unique(bus_ids)

                for bus in buses:
                    bus_msg_ids = msg_ids[bus_ids == bus]

                    unique_ids = sorted(unique(bus_msg_ids).astype("<u8"))

                    bus_map = self.bus_logging_map["CAN"].setdefault(bus, {})

                    for msg_id in unique_ids:
                        bus_map[int(msg_id)] = group_index

            self._set_temporary_master(None)
            group.record = None

        else:

            is_j1939 = dbc.contains_j1939

            if is_j1939:
                messages = {message.arbitration_id.pgn: message for message in dbc}
            else:
                messages = {message.arbitration_id.id: message for message in dbc}

            msg_map = {}

            parents, dtypes = self._prepare_record(group)
            data = self._load_data(group, optimize_read=False)

            for fragment_index, fragment in enumerate(data):
                if dtypes.itemsize:
                    group.record = fromstring(fragment[0], dtype=dtypes)
                else:
                    group.record = None
                    return

                self._set_temporary_master(None)
                self._set_temporary_master(self.get_master(group_index, data=fragment))

                bus_ids = self.get(
                    "CAN_DataFrame.BusChannel",
                    group=group_index,
                    data=fragment,
                    samples_only=True,
                )[0].astype("<u1")

                msg_ids = (
                    self.get(
                        "CAN_DataFrame.ID", group=group_index, data=fragment
                    ).astype("<u4")
                    & 0x1FFFFFFF
                )

                if is_j1939:

                    tmp_pgn = msg_ids.samples >> 8
                    ps = tmp_pgn & 0xFF
                    pf = (msg_ids.samples >> 16) & 0xFF
                    _pgn = tmp_pgn & 0x3FF00
                    msg_ids.samples = where(pf >= 240, _pgn + ps, _pgn)

                data_bytes = self.get(
                    "CAN_DataFrame.DataBytes",
                    group=group_index,
                    data=fragment,
                    samples_only=True,
                )[0]

                buses = unique(bus_ids)
                if len(bus_ids) == 0:
                    continue

                for bus in buses:
                    idx_ = bus_ids == bus
                    bus_msg_ids = msg_ids.samples[idx_]

                    bus_t = msg_ids.timestamps[idx_]
                    bus_data_bytes = data_bytes[idx_]

                    unique_ids = sorted(unique(bus_msg_ids).astype("<u8"))

                    bus_map = self.bus_logging_map["CAN"].setdefault(bus, {})

                    for msg_id in unique_ids:
                        bus_map[int(msg_id)] = group_index

                    for msg_id in unique_ids:
                        message = messages.get(msg_id, None)
                        if message is None:
                            continue

                        idx = bus_msg_ids == msg_id
                        payload = bus_data_bytes[idx]
                        t = bus_t[idx]

                        extracted_signals = extract_mux(
                            payload, message, msg_id, bus, t
                        )

                        for entry, signals in extracted_signals.items():
                            if len(next(iter(signals.values()))["samples"]) == 0:
                                continue
                            if entry not in msg_map:
                                sigs = []

                                for name_, signal in signals.items():
                                    sig = Signal(
                                        samples=signal["samples"],
                                        timestamps=signal["t"],
                                        name=signal["name"],
                                        comment=signal["comment"],
                                        unit=signal["unit"],
                                        invalidation_bits=signal["invalidation_bits"],
                                        display_name=f"CAN{bus}.{message.name}.{signal['name']}",
                                    )

                                    sigs.append(sig)

                                cg_nr = self.append(
                                    sigs,
                                    acq_name=f"from CAN{bus} message ID=0x{msg_id:X}",
                                    comment=f"{message} 0x{msg_id:X}",
                                    common_timebase=True,
                                )

                                msg_map[entry] = cg_nr

                                for ch_index, ch in enumerate(
                                    self.groups[cg_nr].channels
                                ):
                                    if ch_index == 0:
                                        continue

                                    entry = cg_nr, ch_index

                                    name_ = f"{message}.{ch.name}"
                                    self.channels_db.add(name_, entry)

                                    name_ = f"CAN{bus}.{message}.{ch.name}"
                                    self.channels_db.add(name_, entry)

                                    name_ = f"CAN_DataFrame_{msg_id}.{ch.name}"
                                    self.channels_db.add(name_, entry)

                                    name_ = f"CAN{bus}.CAN_DataFrame_{msg_id}.{ch.name}"
                                    self.channels_db.add(name_, entry)

                            else:

                                index = msg_map[entry]

                                sigs = []

                                for name_, signal in signals.items():

                                    sigs.append(
                                        (signal["samples"], signal["invalidation_bits"])
                                    )

                                    t = signal["t"]

                                sigs.insert(0, (t, None))

                                self.extend(index, sigs)
                self._set_temporary_master(None)
                group.record = None

    def _process_lin_logging(self, group_index, grp):

        channels = grp.channels
        group = grp

        dbc = None

        for i, channel in enumerate(channels):
            if channel.name == "LIN_Frame":
                attachment_addr = channel.attachment
                if attachment_addr is not None:
                    if attachment_addr not in self._dbc_cache:

                        attachment, at_name, md5_sum = self.extract_attachment(
                            index=attachment_addr,
                            decryption_function=self._decryption_function,
                        )
                        if at_name.suffix.lower() not in (".arxml", ".dbc", ".ldf"):
                            message = f'Expected .dbc, .arxml or .ldf file as LIN channel attachment but got "{at_name}"'
                            logger.warning(message)
                        elif not attachment:
                            message = f'Attachment "{at_name}" not found'
                            logger.warning(message)
                        else:
                            contents = (
                                None if at_name.suffix.lower() == ".ldf" else attachment
                            )
                            dbc = load_can_database(at_name, contents=contents)
                            if dbc:
                                self._dbc_cache[attachment_addr] = dbc
                    else:
                        dbc = self._dbc_cache[attachment_addr]
                break

        if dbc is None:
            parents, dtypes = self._prepare_record(group)
            data = self._load_data(group, optimize_read=False)

            for fragment_index, fragment in enumerate(data):
                if dtypes.itemsize:
                    group.record = fromstring(fragment[0], dtype=dtypes)
                else:
                    group.record = None

                    return

                self._set_temporary_master(None)
                self._set_temporary_master(self.get_master(group_index, data=fragment))

                msg_ids = (
                    self.get(
                        "LIN_Frame.ID",
                        group=group_index,
                        data=fragment,
                        samples_only=True,
                    )[0].astype("<u4")
                    & 0x1FFFFFFF
                )

                unique_ids = sorted(unique(msg_ids).astype("<u8"))

                lin_map = self.bus_logging_map["LIN"]

                for msg_id in unique_ids:
                    lin_map[int(msg_id)] = group_index

            self._set_temporary_master(None)
            group.record = None

        else:

            messages = {message.arbitration_id.id: message for message in dbc}

            msg_map = {}

            parents, dtypes = self._prepare_record(group)
            data = self._load_data(group, optimize_read=False)

            for fragment_index, fragment in enumerate(data):
                if dtypes.itemsize:
                    group.record = fromstring(fragment[0], dtype=dtypes)
                else:
                    group.record = None
                    return

                self._set_temporary_master(None)
                self._set_temporary_master(self.get_master(group_index, data=fragment))

                msg_ids = (
                    self.get("LIN_Frame.ID", group=group_index, data=fragment).astype(
                        "<u4"
                    )
                    & 0x1FFFFFFF
                )

                data_bytes = self.get(
                    "LIN_Frame.DataBytes",
                    group=group_index,
                    data=fragment,
                    samples_only=True,
                )[0]

                bus_msg_ids = msg_ids.samples

                bus_t = msg_ids.timestamps
                bus_data_bytes = data_bytes

                unique_ids = sorted(unique(bus_msg_ids).astype("<u8"))

                lin_map = self.bus_logging_map["LIN"]

                for msg_id in unique_ids:
                    lin_map[int(msg_id)] = group_index

                for msg_id in unique_ids:
                    message = messages.get(msg_id, None)
                    if message is None:
                        continue

                    idx = bus_msg_ids == msg_id
                    payload = bus_data_bytes[idx]
                    t = bus_t[idx]

                    extracted_signals = extract_mux(payload, message, msg_id, 0, t)

                    for entry, signals in extracted_signals.items():
                        if len(next(iter(signals.values()))["samples"]) == 0:
                            continue
                        if entry not in msg_map:
                            sigs = []

                            for name_, signal in signals.items():
                                sig = Signal(
                                    samples=signal["samples"],
                                    timestamps=signal["t"],
                                    name=signal["name"],
                                    comment=signal["comment"],
                                    unit=signal["unit"],
                                    invalidation_bits=signal["invalidation_bits"],
                                    display_name=f"LIN.{message.name}.{signal['name']}",
                                )

                                sigs.append(sig)

                            cg_nr = self.append(
                                sigs,
                                acq_name=f"from LIN message ID=0x{msg_id:X}",
                                comment=f"{message} 0x{msg_id:X}",
                                common_timebase=True,
                            )

                            msg_map[entry] = cg_nr

                            for ch_index, ch in enumerate(self.groups[cg_nr].channels):
                                if ch_index == 0:
                                    continue

                                entry = cg_nr, ch_index

                                name_ = f"{message}.{ch.name}"
                                self.channels_db.add(name_, entry)

                                name_ = f"LIN.{message}.{ch.name}"
                                self.channels_db.add(name_, entry)

                                name_ = f"LIN_Frame_{msg_id}.{ch.name}"
                                self.channels_db.add(name_, entry)

                                name_ = f"LIN.LIN_Frame_{msg_id}.{ch.name}"
                                self.channels_db.add(name_, entry)

                        else:

                            index = msg_map[entry]

                            sigs = []

                            for name_, signal in signals.items():

                                sigs.append(
                                    (signal["samples"], signal["invalidation_bits"])
                                )

                                t = signal["t"]

                            sigs.insert(0, (t, None))

                            self.extend(index, sigs)
                self._set_temporary_master(None)
                group.record = None

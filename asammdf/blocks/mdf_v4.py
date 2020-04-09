"""
ASAM MDF version 4 file format module
"""

import logging
import xml.etree.ElementTree as ET
import os
import sys
from copy import deepcopy
from collections import defaultdict
from hashlib import md5
from itertools import chain
from math import ceil
from tempfile import TemporaryFile
from zlib import decompress, compress
from pathlib import Path
import mmap
from functools import lru_cache
from time import perf_counter
from lz4.frame import compress as lz_compress
from lz4.frame import decompress as lz_decompress
from traceback import format_exc

from numpy import (
    arange,
    array,
    array_equal,
    concatenate,
    cumsum,
    dtype,
    empty,
    flip,
    float32,
    float64,
    frombuffer,
    linspace,
    nonzero,
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
    fliplr,
    searchsorted,
    full,
    unique,
    column_stack,
    where,
    argwhere,
)

from numpy.core.records import fromarrays, fromstring
from numpy.core.defchararray import encode, decode
import canmatrix
from pandas import DataFrame

from . import v4_constants as v4c
from ..signal import Signal
from .conversion_utils import conversion_transfer
from .finalization_shim import FinalizationShim
from .utils import (
    UINT8_u,
    UINT8_uf,
    UINT16_u,
    UINT16_uf,
    UINT32_u,
    UINT32_uf,
    UINT64_u,
    UINT64_uf,
    CHANNEL_COUNT,
    CONVERT,
    ChannelsDB,
    MdfException,
    SignalSource,
    as_non_byte_sized_signed_int,
    fmt_to_datatype_v4,
    get_fmt_v4,
    UniqueDB,
    debug_channel,
    extract_cncomment_xml,
    validate_version_argument,
    count_channel_groups,
    info_to_datatype_v4,
    is_file_like,
    sanitize_xml,
    Group,
    DataBlockInfo,
    SignalDataBlockInfo,
    InvalidationBlockInfo,
    extract_can_signal,
    load_can_database,
    VirtualChannelGroup,
    all_blocks_addresses,
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
from ..version import __version__


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
    from .cutils import extract, sort_data_block, lengths, get_vlsd_offsets
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
        i = 0
        size = len(signal_data)
        pos = 0
        rem = b''
        while i + record_id_nr < size:
            (rec_id,) = _unpack_stuct(signal_data, i)
            # skip record id
            i += record_id_nr
            rec_size = cg_size[rec_id]
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
        offsets = [0,] + [len(item) for item in data]
        offsets = cumsum(offsets)
        return offsets[:-1], offsets[-1]


class MDF4(object):
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
    * ``param`` - row size used for tranposizition, in case of tranposed zipped blockss


    Parameters
    ----------
    name : string
        mdf file name (if provided it must be a real file name) or
        file-like object

        * if *full* the data group binary data block will be memorised in RAM
        * if *low* the channel data is read from disk on request, and the
          metadata is memorized into RAM
        * if *minimum* only minimal data is memorized into RAM

    version : string
        mdf file version ('4.00', '4.10', '4.11', '4.20'); default '4.10'
    callback : function
        keyword only argument: function to call to update the progress; the
        function must accept two arguments (the current progress and maximum
        progress value)
    use_display_names : bool
        keyword only argument: for MDF4 files parse the XML channel comment to
        search for the display name; XML parsing is quite expensive so setting
        this to *False* can decrease the loading times very much; default
        *False*


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

    def __init__(self, name=None, version="4.10", **kwargs):

        self.groups = []
        self.header = None
        self.identification = None
        self.file_history = []
        self.channels_db = ChannelsDB()
        self.can_logging_db = {}
        self.masters_db = {}
        self.attachments = []
        self._attachments_cache = {}
        self.file_comment = None
        self.events = []

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

        self._tempfile = TemporaryFile()
        self._file = None

        self._read_fragment_size = 0 * 2 ** 20
        self._write_fragment_size = 4 * 2 ** 20
        self._use_display_names = kwargs.get("use_display_names", False)
        self._remove_source_from_channel_names = kwargs.get(
            "remove_source_from_channel_names", False
        )
        self.copy_on_get = kwargs.get("copy_on_get", True)
        self._single_bit_uint_as_bool = False
        self._integer_interpolation = 0
        self.virtual_groups = {} # master group 2 referencing groups
        self.virtual_groups_map = {} # group index 2 master group

        self._master = None

        self.last_call_info = None

        # make sure no appended block has the address 0
        self._tempfile.write(b"\0")

        self._callback = kwargs.get("callback", None)

        if name:
            if is_file_like(name):
                self._file = name
                self.name = Path("From_FileLike.mf4")
                self._from_filelike = True
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
            self.name = Path("new.mf4")

        if self.version >= "4.20":
            self._column_storage = kwargs.get("column_storage", True)
        else:
            self._column_storage = False

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
                addresses = all_blocks_addresses(self._file)
            else:
                addresses = []

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
                )
                self._cg_map[cg_addr] = dg_cntr
                channel_group = grp.channel_group = block

                grp.record_size = cg_size

                if channel_group.flags & v4c.FLAG_CG_VLSD:
                    # VLDS flag
                    record_id = channel_group.record_id
                    cg_size[record_id] = 0
                elif channel_group.flags & v4c.FLAG_CG_BUS_EVENT:
                    if channel_group.acq_source is None:
                        grp.CAN_logging = False
                    else:
                        bus_type = channel_group.acq_source.bus_type
                        if bus_type == v4c.BUS_TYPE_CAN:
                            grp.CAN_logging = True

                        else:
                            # only CAN bus logging is supported
                            grp.CAN_logging = False
                    samples_size = channel_group.samples_byte_nr
                    inval_size = channel_group.invalidation_bytes_nr
                    record_id = channel_group.record_id
                    cg_size[record_id] = samples_size + inval_size
                else:

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
                    total_size += channel_group.samples_byte_nr * channel_group.cycles_nr
                    inval_total_size += (
                        channel_group.invalidation_bytes_nr * channel_group.cycles_nr
                    )
                else:
                    block_type = b"##DT"
                    total_size += (
                        channel_group.samples_byte_nr
                        + channel_group.invalidation_bytes_nr
                    ) * channel_group.cycles_nr

#            if not is_finalised:
#                total_size = None
#                inval_total_size = None

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

        #TODO: attempt finalisation here

        # all channels have been loaded so now we can link the
        # channel dependencies and load the signal data for VLSD channels
        for gp_index, grp in enumerate(self.groups):

            if self.version >= "4.20" and grp.channel_group.flags & v4c.FLAG_CG_REMOTE_MASTER:
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
                            ch_addr = dep[f"output_quantity_ch_addr"]
                            if ch_addr:
                                ref_channel = self._ch_map[ch_addr]
                                dep.output_quantity_channel = ref_channel
                            else:
                                dep.output_quantity_channel = None

                        if dep.flags & v4c.FLAG_CA_COMPARISON_QUANTITY:
                            ch_addr = dep[f"comparison_quantity_ch_addr"]
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
                                        stream=stream, address=cc_addr, mapped=mapped,
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

        for grp in self.groups:
            channels = grp.channels
            if (
                len(channels) == 1
                and channels[0].dtype_fmt.itemsize == grp.channel_group.samples_byte_nr
            ):
                grp.single_channel_dtype = channels[0].dtype_fmt

        self._process_can_logging()

        # append indexes of groups that contain raw CAN bus logging and
        # store signals and metadata that will be used to create the new
        # groups.
        raw_can = []
        processed_can = []
        for i, group in enumerate(self.groups):
            if group.CAN_logging:
                if group.CAN_id not in self.can_logging_db:
                    self.can_logging_db[group.CAN_id] = {}
                message_id = group.message_id
                if message_id is not None:
                    for id_ in message_id:
                        self.can_logging_db[group.CAN_id][id_] = i
            else:
                continue

            if group.raw_can:
                try:
                    _sig = self.get(
                        "CAN_DataFrame", group=i, ignore_invalidation_bits=True
                    )
                except MdfException:
                    continue

                can_ids = Signal(
                    _sig.samples["CAN_DataFrame.ID"] & 0x1fffffff,
                    _sig.timestamps,
                    name="can_ids",
                )

                all_can_ids = unique(can_ids.samples).tolist()

                if all_can_ids:
                    group.message_id = set()
                    for message_id in all_can_ids:
                        if message_id > 0x80000000:
                            message_id -= 0x80000000

                        self.can_logging_db[group.CAN_id][message_id] = i
                        group.message_id.add(message_id)

                payload = _sig.samples["CAN_DataFrame.DataBytes"]

                attachment = _sig.attachment
                if (
                    attachment
                    and attachment[0]
                    and attachment[1].name.lower().endswith(("dbc", "arxml"))
                ):
                    attachment, at_name = attachment

                    db = load_can_database(at_name, attachment)

                    if db:
                        try:
                            board_units = set(bu.name for bu in db.boardUnits)
                        except AttributeError:
                            board_units = set(bu.name for bu in db.ecus)

                        cg_source = group.channel_group.acq_source

                        all_message_info_extracted = True
                        for message_id in all_can_ids:
                            self.can_logging_db[group.CAN_id][message_id] = i
                            sigs = []
                            try:
                                can_msg = db.frameById(message_id)
                            except AttributeError:
                                can_msg = db.frame_by_id(
                                    canmatrix.ArbitrationId(message_id)
                                )

                            if can_msg:
                                for transmitter in can_msg.transmitters:
                                    if transmitter in board_units:
                                        break
                                else:
                                    transmitter = ""
                                message_name = can_msg.name

                                source = SignalSource(
                                    transmitter,
                                    can_msg.name,
                                    "",
                                    v4c.SOURCE_BUS,
                                    v4c.BUS_TYPE_CAN,
                                )

                                idx = nonzero(can_ids.samples == message_id)[0]
                                data = payload[idx]
                                t = can_ids.timestamps[idx].copy()
                                if can_ids.invalidation_bits is not None:
                                    invalidation_bits = can_ids.invalidation_bits[idx]
                                else:
                                    invalidation_bits = None

                                for signal in sorted(
                                    can_msg.signals, key=lambda x: x.name
                                ):

                                    sig_vals = extract_can_signal(signal, data)

                                    # conversion = ChannelConversion(
                                    #     a=float(signal.factor),
                                    #     b=float(signal.offset),
                                    #     conversion_type=v4c.CONVERSION_TYPE_LIN,
                                    # )
                                    # conversion.unit = signal.unit or ""
                                    sigs.append(
                                        Signal(
                                            sig_vals,
                                            t,
                                            name=signal.name,
                                            conversion=None,
                                            source=source,
                                            unit=signal.unit,
                                            raw=False,
                                            invalidation_bits=invalidation_bits,
                                        )
                                    )

                                processed_can.append(
                                    [
                                        sigs,
                                        message_id,
                                        message_name,
                                        cg_source,
                                        group.CAN_id,
                                    ]
                                )
                            else:
                                all_message_info_extracted = False

                        if all_message_info_extracted:
                            raw_can.append(i)

        if processed_can:

            for sigs, message_id, message_name, cg_source, can_id in processed_can:
                self.append(
                    sigs, "Extracted from raw CAN bus logging", common_timebase=True
                )
                group = self.groups[-1]
                group.CAN_database = message_name != ""
                group.CAN_logging = False
                group.CAN_id = can_id
                if message_id > 0:
                    if message_id > 0x80000000:
                        message_id -= 0x80000000
                        group.extended_id = True
                    else:
                        group.extended_id = False
                    group.message_name = message_name
                    group.message_id = {message_id}
                group.channel_group.acq_source = cg_source
                group.data_group.comment = (
                    f'From message {hex(message_id)}="{message_name}"'
                )

        self.can_logging_db = {}

        for i, group in enumerate(self.groups):
            if not group.CAN_logging:
                continue
            if not group.CAN_id in self.can_logging_db:
                self.can_logging_db[group.CAN_id] = {}
            if group.message_id is not None:
                for id_ in group.message_id:
                    self.can_logging_db[group.CAN_id][id_] = i

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

            channel = Channel(
                address=ch_addr,
                stream=stream,
                cc_map=self._cc_map,
                si_map=self._si_map,
                at_map=self._attachments_map,
                use_display_names=self._use_display_names,
                mapped=mapped,
                tx_map=self._interned_strings,
            )

            if self._remove_source_from_channel_names:
                channel.name = channel.name.split(path_separator, 1)[0]

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
                    ca_block = ChannelArrayBlock(
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

    def _process_can_logging(self):
        for i, grp in enumerate(self.groups):
            if not grp.CAN_logging:
                continue

            channel_group = grp.channel_group
            message_name = channel_group.acq_name
            dg_cntr = i
            channels = grp.channels
            neg_ch_cntr = -1

            comment = channel_group.acq_source.comment
            if comment:
                try:
                    comment_xml = ET.fromstring(comment)
                except ET.ParseError as e:
                    logger.error(f"could not parse acq_source comment; {e}")
                else:
                    common_properties = comment_xml.find(".//common_properties")
                    grp.CAN_id = next(
                        (
                            f"CAN{e.text}"
                            for e in common_properties
                            if e.get("name") == "ChannelNo"
                        ),
                        None,
                    )

            if grp.CAN_id:
                if message_name == "CAN_DataFrame":
                    # this is a raw CAN bus logging channel group
                    # it will be later processed to extract all
                    # signals to new groups (one group per CAN message)
                    grp.raw_can = True

                elif message_name in ("CAN_ErrorFrame", "CAN_RemoteFrame"):
                    # for now ignore bus logging flag
                    pass
                else:
                    comment = channel_group.comment
                    if comment:

                        comment_xml = ET.fromstring(sanitize_xml(comment))
                        can_msg_type = comment_xml.find(".//TX").text
                        if can_msg_type is not None:
                            can_msg_type = can_msg_type.strip(" \t\r\n")
                        else:
                            can_msg_type = "CAN_DataFrame"
                        if can_msg_type == "CAN_DataFrame":
                            common_properties = comment_xml.find(".//common_properties")
                            message_id = -1
                            for e in common_properties:
                                name = e.get("name")
                                if name == "MessageID":
                                    message_id = int(e.text)
                                    break

                            if message_id > 0:
                                if message_id > 0x80000000:
                                    message_id -= 0x80000000
                                    grp.extended_id = True
                                grp.message_name = message_name
                                grp.message_id = {message_id}

                        else:
                            grp.raw_can = True
                            grp.CAN_logging = False
                            message = (
                                f"Invalid bus logging channel group metadata: {comment}"
                            )
                            logger.warning(message)
                    else:
                        grp.raw_can = True

            else:
                try:
                    data = self._load_data(self.groups[i])
                    for fragment in data:
                        bus_ids = self.get(
                            "CAN_DataFrame.BusChannel",
                            group=i,
                            data=fragment,
                            samples_only=True,
                        )[0]

                        can_ids = self.get(
                            "CAN_DataFrame.ID",
                            group=i,
                            data=fragment,
                            samples_only=True,
                        )[0] & 0x1fffffff

                        if len(bus_ids):

                            for can_id, message_id in unique(
                                column_stack((bus_ids.astype(int), can_ids)), axis=0
                            ):
                                can_id = f"CAN{can_id}"
                                if can_id not in self.can_logging_db:
                                    self.can_logging_db[can_id] = {}
                                grp.CAN_id = can_id
                                self.can_logging_db[can_id][message_id] = i

                        if grp.message_id is None:
                            grp.message_id = set(unique(can_ids))
                        else:
                            grp.message_id = grp.message_id | set(unique(can_ids))

                except MdfException:
                    grp.CAN_logging = False
                    pass

            if (
                grp.CAN_id is not None
                and grp.message_id is not None
                and len(grp.message_id) == 1
            ):
                for j, dep in enumerate(grp.channel_dependencies):
                    if dep:
                        channel = grp.channels[j]
                        break
                try:
                    addr = channel.attachment_addr
                except AttributeError:
                    addr = 0
                if addr:
                    attachment_addr = self._attachments_map[addr]
                    if attachment_addr not in self._dbc_cache:
                        attachment, at_name = self.extract_attachment(
                            index=attachment_addr
                        )
                        if not at_name.name.lower().endswith(("dbc", "arxml")):
                            message = f'Expected .dbc or .arxml file as CAN channel attachment but got "{at_name}"'
                            logger.warning(message)
                            grp.CAN_database = False
                        elif not attachment:
                            message = f'Attachment "{at_name}" not found'
                            logger.warning(message)
                            grp.CAN_database = False
                        else:

                            dbc = load_can_database(at_name, attachment)
                            if dbc is None:
                                grp.CAN_database = False
                            else:
                                self._dbc_cache[attachment_addr] = dbc
                                grp.CAN_database = True

                    else:
                        grp.CAN_database = True
                else:
                    grp.CAN_database = False

                if grp.CAN_database:

                    # here we make available multiple ways to refer to
                    # CAN signals by using fake negative indexes for
                    # the channel entries in the channels_db

                    grp.dbc_addr = attachment_addr

                    message_id = next(iter(grp.message_id))
                    message_name = grp.message_name
                    can_id = grp.CAN_id

                    try:
                        can_msg = self._dbc_cache[attachment_addr].frameById(message_id)
                    except AttributeError:
                        can_msg = self._dbc_cache[attachment_addr].frame_by_id(
                            canmatrix.ArbitrationId(message_id)
                        )

                    if can_msg:
                        can_msg_name = can_msg.name

                        for entry in self.channels_db["CAN_DataFrame.DataBytes"]:
                            if entry[0] == dg_cntr:
                                index = entry[1]
                                break

                        payload = channels[index]

                        logging_channels = grp.logging_channels

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
                            bit_offset = signal.start_bit % 8
                            byte_offset = signal.start_bit // 8
                            bit_count = signal.size
                            comment = signal.comment or ""

                            if (signal.factor, signal.offset) != (1, 0):
                                conversion = ChannelConversion(
                                    a=float(signal.factor),
                                    b=float(signal.offset),
                                    conversion_type=v4c.CONVERSION_TYPE_LIN,
                                )
                                conversion.unit = signal.unit or ""
                            else:
                                conversion = None

                            kwargs = {
                                "channel_type": v4c.CHANNEL_TYPE_VALUE,
                                "data_type": s_type,
                                "sync_type": payload.sync_type,
                                "byte_offset": byte_offset + payload.byte_offset,
                                "bit_offset": bit_offset,
                                "bit_count": bit_count,
                                "min_raw_value": 0,
                                "max_raw_value": 0,
                                "lower_limit": 0,
                                "upper_limit": 0,
                                "flags": 0,
                                "pos_invalidation_bit": payload.pos_invalidation_bit,
                            }

                            log_channel = Channel(**kwargs)
                            log_channel.name = name_
                            log_channel.comment = comment
                            log_channel.source = deepcopy(channel.source)
                            log_channel.conversion = conversion
                            log_channel.unit = signal.unit or ""

                            logging_channels.append(log_channel)

                            entry = dg_cntr, neg_ch_cntr
                            self.channels_db.add(name_, entry)

                            name_ = f"{message_name}.{signal_name}"
                            self.channels_db.add(name_, entry)

                            name_ = f"CAN{can_id}.{message_name}.{signal_name}"
                            self.channels_db.add(name_, entry)

                            name_ = f"{can_msg_name}.{signal_name}"
                            self.channels_db.add(name_, entry)

                            name_ = f"CAN{can_id}.{can_msg_name}.{signal_name}"
                            self.channels_db.add(name_, entry)

                            neg_ch_cntr -= 1

                        grp.channel_group["flags"] &= ~v4c.FLAG_CG_PLAIN_BUS_EVENT

    def _load_signal_data(
        self, address=None, stream=None, group=None, index=None, offset=0, count=None
    ):
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

        with_bounds = False

        if address == 0:
            data = b""

        elif address is not None and stream is not None:
            stream.seek(address)
            blk_id = stream.read(4)
            if blk_id == b"##SD":
                data = DataBlock(address=address, stream=stream)
                data = data.data
            elif blk_id == b"##DZ":
                data = DataZippedBlock(address=address, stream=stream)
                data = data.data
            elif blk_id == b"##CG":
                group = self.groups[self._cg_map[address]]
                data = b"".join(fragment[0] for fragment in self._load_data(group))
            elif blk_id == b"##DL":
                data = []
                while address:
                    # the data list will contain only links to SDBLOCK's
                    data_list = DataList(address=address, stream=stream)
                    nr = data_list.links_nr
                    # aggregate data from all SDBLOCK
                    for i in range(nr - 1):
                        addr = data_list[f"data_block_addr{i}"]
                        stream.seek(addr)
                        blk_id = stream.read(4)
                        if blk_id == b"##SD":
                            block = DataBlock(address=addr, stream=stream)
                            data.append(block.data)
                        elif blk_id == b"##DZ":
                            block = DataZippedBlock(address=addr, stream=stream)
                            data.append(block.data)
                        else:
                            message = f'Expected SD, DZ or DL block at {hex(address)} but found id="{blk_id}"'
                            logger.warning(message)
                            return b"", with_bounds
                    address = data_list.next_dl_addr
                data = b"".join(data)
            elif blk_id == b"##CN":
                data = b""
            elif blk_id == b"##HL":
                hl = HeaderList(address=address, stream=stream)

                data, with_bounds = self._load_signal_data(
                    address=hl.first_dl_addr, stream=stream, group=group, index=index
                )
            elif blk_id == b"##AT":
                data = b""
            else:
                message = f'Expected AT, CG, SD, DL, DZ or CN block at {hex(address)} but found id="{blk_id}"'
                logger.warning(message)
                data = b""

        elif group is not None and index is not None:
            if group.data_location == v4c.LOCATION_ORIGINAL_FILE:
                data, with_bounds = self._load_signal_data(
                    address=group.signal_data[index], stream=self._file
                )
            elif group.data_location == v4c.LOCATION_MEMORY:
                data = group.signal_data[index]
            else:
                data = []
                stream = self._tempfile
                address = group.signal_data[index]

                if address:
                    if isinstance(address, int):

                        if address in self._cg_map:
                            group = self.groups[self._cg_map[address]]
                            data.append(b"".join(e[0] for e in self._load_data(group)))

                    else:
                        if isinstance(address[0], SignalDataBlockInfo):

                            if address[0].offsets is not None:
                                with_bounds = True

                                current_offset = 0
                                if count is not None:
                                    end = offset + count
                                else:
                                    end = None

                                for info in address:

                                    current_count = info.count

                                    if current_offset + current_count < offset:
                                        current_offset += current_count
                                        continue

                                    if current_offset < offset:
                                        start_addr = (
                                            info.address
                                            + info.offsets[offset - current_offset]
                                        )
                                    else:
                                        start_addr = info.address

                                    if end is not None:
                                        if end <= current_offset:
                                            break
                                        elif end >= current_offset + current_count:
                                            end_addr = info.address + info.size
                                        else:
                                            end_addr = (
                                                info.address
                                                + info.offsets[end - current_offset]
                                            )
                                    else:
                                        end_addr = info.address + info.size

                                    size = int(end_addr - start_addr)
                                    start_addr = int(start_addr)

                                    stream.seek(start_addr)
                                    data.append(stream.read(size))
                                    current_offset += current_count

                            else:
                                for info in address:
                                    if not info.size:
                                        continue
                                    stream.seek(info.address)
                                    data.append(stream.read(info.size))

                        elif address[0] in self._cg_map:
                            group = self.groups[self._cg_map[address[0]]]
                            data.append(b"".join(e[0] for e in self._load_data(group)))

                data = b"".join(data)
        else:
            data = b""

        return data, with_bounds

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

            if not group.sorted:
                cg_size = group.record_size
                record_id = channel_group.record_id
                record_id_nr = data_group.record_id_len

                if record_id_nr == 1:
                    _unpack_stuct = UINT8_u
                elif record_id_nr == 2:
                    _unpack_stuct = UINT16_u
                elif record_id_nr == 4:
                    _unpack_stuct = UINT32_u
                elif record_id_nr == 8:
                    _unpack_stuct = UINT64_u
                else:
                    message = f"invalid record id size {record_id_nr}"
                    raise MdfException(message)

            blocks = iter(group.data_blocks)

            if group.data_blocks:

                cur_size = 0
                data = []

                cur_invalidation_size = 0
                invalidation_data = []

                while True:
                    try:
                        info = next(blocks)
                        address, size, block_size, block_type, param, block_limit = (
                            info.address,
                            info.raw_size,
                            info.size,
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

                    if group.sorted:
                        if offset + size < record_offset + 1:
                            offset += size
                            if rm and invalidation_size:
                                if invalidation_info.all_valid:
                                    count = size // samples_size
                                    invalidation_offset += count * invalidation_size
                                else:
                                    invalidation_offset += invalidation_info.raw_size
                            continue

                    seek(address)
                    new_data = read(block_size)
                    if block_type == v4c.DZ_BLOCK_DEFLATE:
                        new_data = decompress(new_data, 0, size)
                    elif block_type == v4c.DZ_BLOCK_TRANSPOSED:
                        new_data = decompress(new_data, 0, size)
                        cols = param
                        lines = size // cols

                        nd = frombuffer(new_data[: lines * cols], dtype=uint8)
                        nd = nd.reshape((cols, lines))
                        new_data = nd.T.tostring() + new_data[lines * cols :]
                    elif block_type == v4c.DZ_BLOCK_LZ:
                        new_data = lz_decompress(new_data)

                    if block_limit is not None:
                        new_data = new_data[:block_limit]

                    if not group.sorted:
                        rec_data = []

                        i = 0
                        size = len(new_data)
                        while i < size:
                            (rec_id,) = _unpack_stuct(new_data[i : i + record_id_nr])
                            # skip record id
                            i += record_id_nr
                            rec_size = cg_size[rec_id]
                            if rec_size:
                                endpoint = i + rec_size
                                if rec_id == record_id:
                                    rec_data.append(new_data[i:endpoint])
                                i = endpoint
                            else:
                                (rec_size,) = UINT32_u(new_data[i : i + 4])
                                endpoint = i + rec_size + 4
                                if rec_id == record_id:
                                    rec_data.append(new_data[i:endpoint])
                                i = endpoint
                        new_data = b"".join(rec_data)

                        size = len(new_data)

                    if rm and invalidation_size:

                        if invalidation_info.all_valid:
                            count = size // samples_size
                            new_invalidation_data = bytes(count * invalidation_size)

                        else:
                            seek(invalidation_info.address)
                            new_invalidation_data = read(invalidation_info.size)
                            if invalidation_info.block_type == v4c.DZ_BLOCK_DEFLATE:
                                new_invalidation_data = decompress(
                                    new_invalidation_data,
                                    0,
                                    invalidation_info.raw_size,
                                )
                            elif (
                                invalidation_info.block_type == v4c.DZ_BLOCK_TRANSPOSED
                            ):
                                new_invalidation_data = decompress(
                                    new_invalidation_data,
                                    0,
                                    invalidation_info.raw_size,
                                )
                                cols = invalidation_info.param
                                lines = invalidation_info.raw_size // cols

                                nd = frombuffer(
                                    new_invalidation_data[: lines * cols], dtype=uint8
                                )
                                nd = nd.reshape((cols, lines))
                                new_invalidation_data = (
                                    nd.T.tostring()
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
                        size -= delta
                        offset = record_offset

                        if rm and invalidation_size:
                            delta = invalidation_record_offset - invalidation_offset
                            new_invalidation_data = new_invalidation_data[delta:]
                            inv_size -= delta
                            invalidation_offset = invalidation_record_offset

                    while size >= split_size - cur_size:
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
                        size -= split_size - cur_size
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

                    if size:
                        data.append(new_data)
                        cur_size += size
                        size = 0

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

        parents, dtypes = group.parents, group.types

        if parents is None:
            no_parent = None, None
            channel_group = group.channel_group
            channels = group.channels

            bus_event = channel_group.flags & v4c.FLAG_CG_BUS_EVENT

            record_size = channel_group.samples_byte_nr
            invalidation_bytes_nr = channel_group.invalidation_bytes_nr
            next_byte_aligned_position = 0
            types = []
            current_parent = ""
            parent_start_offset = 0
            parents = {}
            group_channels = UniqueDB()

            neg_index = -1

            sortedchannels = sorted(enumerate(channels), key=lambda i: i[1])
            for original_index, new_ch in sortedchannels:
                start_offset = new_ch.byte_offset
                bit_offset = new_ch.bit_offset
                data_type = new_ch.data_type
                bit_count = new_ch.bit_count
                ch_type = new_ch.channel_type
                dependency_list = group.channel_dependencies[original_index]
                name = new_ch.name

                # handle multiple occurance of same channel name
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

                            byte_size, rem = divmod(size, 8)
                            if rem:
                                byte_size += 1
                            bit_size = byte_size * 8

                            if data_type in(v4c.DATA_TYPE_SIGNED_MOTOROLA, v4c.DATA_TYPE_UNSIGNED_MOTOROLA):
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
                            if isinstance(dependency_list[0], ChannelArrayBlock):
                                ca_block = dependency_list[0]

                                # check if there are byte gaps in the record
                                gap = start_offset - next_byte_aligned_position
                                if gap:
                                    dtype_pair = "", f"V{gap}"
                                    types.append(dtype_pair)

                                size = bit_count // 8 or 1
                                shape = tuple(
                                    ca_block[f"dim_size_{i}"]
                                    for i in range(ca_block.dims)
                                )

                                if (
                                    ca_block.byte_offset_base // size > 1
                                    and len(shape) == 1
                                ):
                                    shape += (ca_block.byte_offset_base // size,)
                                dim = 1
                                for d in shape:
                                    dim *= d

                                if not new_ch.dtype_fmt:
                                    new_ch.dtype_fmt = get_fmt_v4(data_type, bit_count)
                                dtype_pair = (name, new_ch.dtype_fmt, shape)
                                types.append(dtype_pair)

                                current_parent = name
                                next_byte_aligned_position = start_offset + size * dim
                                parents[original_index] = name, 0

                            else:
                                parents[original_index] = no_parent
                                if bus_event:
                                    for logging_channel in group.logging_channels:
                                        parents[neg_index] = (
                                            "CAN_DataFrame.DataBytes",
                                            logging_channel.bit_offset,
                                        )
                                        neg_index -= 1

                    # virtual channels do not have bytes in the record
                    else:
                        parents[original_index] = no_parent

                else:
                    size = bit_offset + bit_count
                    byte_size, rem = divmod(size, 8)
                    if rem:
                        byte_size += 1

                    max_overlapping_size = (
                        next_byte_aligned_position - start_offset
                    ) * 8
                    needed_size = bit_offset + bit_count
                    if max_overlapping_size >= needed_size:
                        if data_type in(v4c.DATA_TYPE_SIGNED_MOTOROLA, v4c.DATA_TYPE_UNSIGNED_MOTOROLA):
                            parents[original_index] = (
                                current_parent,
                                (next_byte_aligned_position - start_offset - byte_size) * 8 + bit_offset,
                            )
                        else:
                            parents[original_index] = (
                                current_parent,
                                ((start_offset - parent_start_offset) * 8) + bit_offset,
                            )
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

    @lru_cache(maxsize=1024)
    def _validate_channel_selection(
        self, name=None, group=None, index=None, source=None
    ):
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
                    "Invalid arguments for channel selection: "
                    'must give "name" or, "group" and "index"'
                )
                raise MdfException(message)
            else:
                gp_nr, ch_nr = group, index
                if ch_nr >= 0:
                    try:
                        grp = self.groups[gp_nr]
                    except IndexError:
                        raise MdfException("Group index out of range")

                    try:
                        grp.channels[ch_nr]
                    except IndexError:
                        raise MdfException(f"Channel index out of range: {(name, group, index)}")
        else:
            if name not in self.channels_db:
                raise MdfException(f'Channel "{name}" not found')
            else:
                if source is not None:
                    for gp_nr, ch_nr in self.channels_db[name]:
                        if source in self._get_source_name(gp_nr, ch_nr):
                            break
                    else:
                        raise MdfException(f"{name} with source {source} not found")
                elif group is None:

                    gp_nr, ch_nr = self.channels_db[name][0]
                    if len(self.channels_db[name]) > 1 and not suppress:
                        message = (
                            f'Multiple occurances for channel "{name}". '
                            f"Using first occurance from data group {gp_nr}. "
                            'Provide both "group" and "index" arguments'
                            " to select another data group"
                        )
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
                                message = f'Channel "{name}" not found in group {group}'
                            else:
                                message = f'Channel "{name}" not found in group {group} at index {index}'
                            raise MdfException(message)

        return gp_nr, ch_nr

    def _get_source_name(self, group, index):
        source = self.groups[group].channels[index].source
        cn_source = source.name if source else ""

        source = self.groups[group].channel_group.acq_source
        cg_source = source.name if source else ""

        return (cn_source, cg_source)

    def _set_temporary_master(self, master):
        self._master = master

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
                id_string, _1, block_len = COMMON_SHORT_uf(stream, address)

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
                                raw_size=size,
                                size=size,
                                param=0,
                                block_limit=block_limit,
                            )
                        )
                # or a DataZippedBlock
                elif id_string == b"##DZ":
                    (
                        _1,
                        _2,
                        _3,
                        _4,
                        original_type,
                        zip_type,
                        _5,
                        param,
                        original_size,
                        zip_size,
                    ) = v4c.DZ_COMMON_uf(stream, address)

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
                                raw_size=original_size,
                                size=zip_size,
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

                            id_string, _1, block_len = COMMON_SHORT_uf(stream, addr)
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
                                            raw_size=size,
                                            size=size,
                                            param=0,
                                            block_limit=block_limit,
                                        )
                                    )
                            # or a DataZippedBlock
                            elif id_string == b"##DZ":
                                (
                                    _1,
                                    _2,
                                    _3,
                                    _4,
                                    original_type,
                                    zip_type,
                                    _5,
                                    param,
                                    original_size,
                                    zip_size,
                                ) = v4c.DZ_COMMON_uf(stream, addr)

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
                                            raw_size=original_size,
                                            size=zip_size,
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

                            id_string, _1, block_len = COMMON_SHORT_uf(stream, addr)
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
                                            raw_size=size,
                                            size=size,
                                            param=0,
                                            block_limit=block_limit,
                                        )
                                    )
                            # or a DataZippedBlock
                            elif id_string == b"##DZ":
                                (
                                    _1,
                                    _2,
                                    _3,
                                    _4,
                                    original_type,
                                    zip_type,
                                    _5,
                                    param,
                                    original_size,
                                    zip_size,
                                ) = v4c.DZ_COMMON_uf(stream, addr)

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
                                            raw_size=original_size,
                                            size=zip_size,
                                            param=param,
                                            block_limit=block_limit,
                                        )
                                    )

                            if has_invalidation:
                                inval_addr = ld[f"invalidation_bits_addr_{i}"]
                                if inval_addr:
                                    id_string, _1, block_len = COMMON_SHORT_uf(
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
                                                raw_size=size,
                                                size=size,
                                                param=0,
                                                block_limit=block_limit,
                                            )
                                    else:
                                        (
                                            _1,
                                            _2,
                                            _3,
                                            _4,
                                            original_type,
                                            zip_type,
                                            _5,
                                            param,
                                            original_size,
                                            zip_size,
                                        ) = v4c.DZ_COMMON_uf(stream, inval_addr)

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
                                                raw_size=original_size,
                                                size=zip_size,
                                                param=param,
                                                block_limit=block_limit,
                                            )
                                else:
                                    info[-1].invalidation_block = InvalidationBlockInfo(
                                        address=0,
                                        block_type=v4c.DT_BLOCK,
                                        raw_size=None,
                                        size=None,
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
                id_string, _1, block_len = COMMON_SHORT_u(
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
                                address=address + COMMON_SIZE,
                                block_type=v4c.DT_BLOCK,
                                raw_size=size,
                                size=size,
                                param=0,
                                block_limit=block_limit,
                            )
                        )
                # or a DataZippedBlock
                elif id_string == b"##DZ":
                    stream.seek(address)
                    (
                        _1,
                        _2,
                        _3,
                        _4,
                        original_type,
                        zip_type,
                        _5,
                        param,
                        original_size,
                        zip_size,
                    ) = v4c.DZ_COMMON_u(stream.read(v4c.DZ_COMMON_SIZE))

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
                                raw_size=original_size,
                                size=zip_size,
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
                            id_string, _, block_len = COMMON_SHORT_u(
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
                                            raw_size=size,
                                            size=size,
                                            param=0,
                                            block_limit=block_limit,
                                        )
                                    )
                            # or a DataZippedBlock
                            elif id_string == b"##DZ":
                                stream.seek(addr)
                                (
                                    _1,
                                    _2,
                                    _3,
                                    _4,
                                    original_type,
                                    zip_type,
                                    _5,
                                    param,
                                    original_size,
                                    zip_size,
                                ) = v4c.DZ_COMMON_u(stream.read(v4c.DZ_COMMON_SIZE))

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
                                            raw_size=original_size,
                                            size=zip_size,
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
                            id_string, _, block_len = COMMON_SHORT_u(
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
                                            raw_size=size,
                                            size=size,
                                            param=0,
                                            block_limit=block_limit,
                                        )
                                    )
                            # or a DataZippedBlock
                            elif id_string == b"##DZ":
                                stream.seek(addr)
                                (
                                    _1,
                                    _2,
                                    _3,
                                    _4,
                                    original_type,
                                    zip_type,
                                    _5,
                                    param,
                                    original_size,
                                    zip_size,
                                ) = v4c.DZ_COMMON_u(stream.read(v4c.DZ_COMMON_SIZE))

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
                                            raw_size=original_size,
                                            size=zip_size,
                                            param=param,
                                            block_limit=block_limit,
                                        )
                                    )

                            if has_invalidation:
                                inval_addr = ld[f"invalidation_bits_addr_{i}"]
                                if inval_addr:
                                    stream.seek(inval_addr)
                                    id_string, _1, block_len = COMMON_SHORT_u(
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
                                                raw_size=size,
                                                size=size,
                                                param=0,
                                                block_limit=block_limit,
                                            )
                                    else:
                                        (
                                            _1,
                                            _2,
                                            _3,
                                            _4,
                                            original_type,
                                            zip_type,
                                            _5,
                                            param,
                                            original_size,
                                            zip_size,
                                        ) = v4c.DZ_COMMON_u(
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
                                                raw_size=original_size,
                                                size=zip_size,
                                                param=param,
                                                block_limit=block_limit,
                                            )
                                else:
                                    info[-1].invalidation_block = InvalidationBlockInfo(
                                        address=0,
                                        block_type=v4c.DT_BLOCK,
                                        raw_size=0,
                                        size=0,
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

    def get_invalidation_bits(self, group_index, channel, fragment):
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
        pos_byte, pos_offset = divmod(ch_invalidation_pos, 8)

        mask = 1 << pos_offset

        invalidation_bits = invalidation[:, pos_byte] & mask
        invalidation_bits = invalidation_bits.astype(bool)

        return invalidation_bits

    def configure(
        self,
        *,
        read_fragment_size=None,
        write_fragment_size=None,
        use_display_names=None,
        single_bit_uint_as_bool=None,
        integer_interpolation=None,
        copy_on_get=None,
    ):
        """ configure MDF parameters

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
        copy_on_get : bool
            copy arrays in the get method

        """

        if read_fragment_size is not None:
            self._read_fragment_size = int(read_fragment_size)

        if write_fragment_size:
            self._write_fragment_size = min(int(write_fragment_size), 4 * 2 ** 20)

        if use_display_names is not None:
            self._use_display_names = bool(use_display_names)

        if single_bit_uint_as_bool is not None:
            self._single_bit_uint_as_bool = bool(single_bit_uint_as_bool)

        if integer_interpolation in (0, 1):
            self._integer_interpolation = int(integer_interpolation)

        if copy_on_get is not None:
            self.copy_on_get = copy_on_get

    def append(self, signals, source_info="Python", common_timebase=False, units=None):
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
        source_info : str
            source information; default 'Python'
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
        >>> mdf.append([s1, s2, s3], 'created by asammdf v4.0.0')
        >>> # case 2: VTAB conversions from channels inside another file
        >>> mdf1 = MDF4('in.mf4')
        >>> ch1 = mdf1.get("Channel1_VTAB")
        >>> ch2 = mdf1.get("Channel2_VTABR")
        >>> sigs = [ch1, ch2]
        >>> mdf2 = MDF4('out.mf4')
        >>> mdf2.append(sigs, 'created by asammdf v4.0.0')
        >>> mdf2.append(ch1, 'just a single channel')
        >>> df = pd.DataFrame.from_dict({'s1': np.array([1, 2, 3, 4, 5]), 's2': np.array([-1, -2, -3, -4, -5])})
        >>> units = {'s1': 'V', 's2': 'A'}
        >>> mdf2.append(df, units=units)

        """
        if isinstance(signals, Signal):
            signals = [signals]
        elif isinstance(signals, DataFrame):
            self._append_dataframe(signals, source_info, units=units)
            return

        if not signals:
            return

        source_block = SourceInformation()
        source_block.name = source_block.path = source_info

        interp_mode = self._integer_interpolation

        prepare_record = True

        # check if the signals have a common timebase
        # if not interpolate the signals using the union of all timbases
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
                        s.interp(t, interpolation_mode=interp_mode) for s in signals
                    ]
                    times = None
                else:
                    t = t_
            else:
                t = t_
        else:
            t = []

        if self.version >= "4.20" and (self._column_storage or 1):
            return self._append_column_oriented(signals, source_block)

        dg_cntr = len(self.groups)

        gp = Group(None)
        gp.signal_data = gp_sdata = []
        gp.signal_data_size = gp_sdata_size = []
        gp.channels = gp_channels = []
        gp.channel_dependencies = gp_dep = []
        gp.signal_types = gp_sig_types = []
        gp.logging_channels = []

        cycles_nr = len(t)

        # channel group
        kwargs = {"cycles_nr": cycles_nr, "samples_byte_nr": 0}
        gp.channel_group = ChannelGroup(**kwargs)
        gp.channel_group.acq_name = source_info

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
            gp_sdata_size.append(0)
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

                if sig_dtype.kind == "u" and signal.bit_count <= 4:
                    s_size = signal.bit_count

                if signal.stream_sync:
                    channel_type = v4c.CHANNEL_TYPE_SYNC
                    if signal.attachment:
                        at_data, at_name = signal.attachment
                        attachment_addr = self.attach(
                            at_data, at_name, mime="video/avi", embedded=False
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
                ch.dtype_fmt = dtype(f"{sig_shape[1:]}{sig_dtype}")

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
                gp_sdata_size.append(0)
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

                    vals = signal.samples.tostring()

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
                    vals = fromarrays(vals).tostring()

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
                gp_sdata_size.append(0)

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
                    gp_sdata_size.append(0)
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
                        size=data_size,
                        count=len(data),
                        offsets=offsets,
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

        signals = None
        del signals

        size = len(samples) * samples.itemsize

        if size:
            if self.version < "4.20":

                data_address = self._tempfile.tell()

                data = samples.tobytes()
                raw_size = len(data)
                data = lz_compress(data)

                size = len(data)
                self._tempfile.write(data)

                gp.data_blocks.append(
                    DataBlockInfo(
                        address=data_address,
                        block_type=v4c.DZ_BLOCK_LZ,
                        raw_size=raw_size,
                        size=size,
                        param=0,
                    )
                )

            else:
                data_address = self._tempfile.tell()
                gp.uses_ld = True
                data_address = tell()

                data = samples.tobytes()
                raw_size = len(data)
                data = lz_compress(data)

                size = len(data)
                self._tempfile.write(data)

                gp.data_blocks.append(
                    DataBlockInfo(
                        address=data_address,
                        block_type=v4c.DZ_BLOCK_LZ,
                        raw_size=raw_size,
                        size=size,
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
                            raw_size=raw_size,
                            size=size,
                            param=None,
                        )
                    )
        gp.data_location = v4c.LOCATION_TEMPORARY_FILE

        return dg_cntr

    def _append_column_oriented(self, signals, source_block):
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
        gp.signal_data_size = gp_sdata_size = []
        gp.channels = gp_channels = []
        gp.channel_dependencies = gp_dep = []
        gp.signal_types = gp_sig_types = []
        gp.logging_channels = []
        gp.uses_ld = True
        gp.data_group = DataGroup()
        gp.sorted = True

        samples = signals[0].timestamps

        cycles_nr = len(samples)

        # channel group
        kwargs = {"cycles_nr": cycles_nr, "samples_byte_nr": 0}
        gp.channel_group = ChannelGroup(**kwargs)
        gp.channel_group.acq_name = source_block.name

        self.groups.append(gp)

        ch_cntr = 0
        types = []
        parents = {}
        ch_cntr = 0
        offset = 0

        prepare_record = True

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
        gp_sdata_size.append(0)
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
                            raw_size=chunk,
                            size=chunk,
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
                            raw_size=size,
                            size=size,
                            param=0,
                        )
                    )
                    size = 0
        else:
            gp.data_location = v4c.LOCATION_TEMPORARY_FILE

        for signal in signals:
            gp = Group(None)
            gp.signal_data = gp_sdata = []
            gp.signal_data_size = gp_sdata_size = []
            gp.channels = gp_channels = []
            gp.channel_dependencies = gp_dep = []
            gp.signal_types = gp_sig_types = []
            gp.logging_channels = []
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
            gp.channel_group.acq_name = source_block.name
            gp.channel_group.acq_source = source_block
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
                        at_data, at_name = signal.attachment
                        attachment_addr = self.attach(
                            at_data, at_name, mime="video/avi", embedded=False
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
                gp_sdata_size.append(0)
                entry = (dg_cntr, ch_cntr)
                self.channels_db.add(name, entry)
                if ch.display_name:
                    self.channels_db.add(ch.display_name, entry)

                # update the parents as well
                parents[ch_cntr] = name, 0

                _shape = sig_shape[1:]
                types.append((name, sig_dtype, _shape))
                gp.single_channel_dtype = ch.dtype_fmt = dtype(f"{_shape}{sig_dtype}")

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
                gp_sdata_size.append(0)

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
                offset = gp['types'].itemsize

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
                    gp_sdata_size.append(0)
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
                        size=data_size,
                        count=len(data),
                        offsets=offsets,
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
                        raw_size=raw_size,
                        size=size,
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
                            raw_size=raw_size,
                            size=size,
                            param=None,
                        )
                    )

            gp.data_location = v4c.LOCATION_TEMPORARY_FILE

        return initial_dg_cntr

    def _append_dataframe(self, df, source_info="", units=None):
        """
        Appends a new data group from a Pandas data frame.

        """

        units = units or {}

        t = df.index
        index_name = df.index.name
        time_name = index_name or "time"
        sync_type = v4c.SYNC_TYPE_TIME
        time_unit = "s"

        dg_cntr = len(self.groups)

        gp = Group(None)
        gp.signal_data = gp_sdata = []
        gp.signal_data_size = gp_sdata_size = []
        gp.channels = gp_channels = []
        gp.channel_dependencies = gp_dep = []
        gp.signal_types = gp_sig_types = []
        gp.logging_channels = []

        cycles_nr = len(t)

        # channel group
        kwargs = {"cycles_nr": cycles_nr, "samples_byte_nr": 0}
        gp.channel_group = ChannelGroup(**kwargs)
        gp.channel_group.acq_name = source_info

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

        source_block = SourceInformation()
        source_block.name = source_block.path = source_info

        if df.shape[0]:
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
            ch.source = source_block
            ch.dtype_fmt = t.dtype
            name = time_name
            gp_channels.append(ch)

            gp_sdata.append(None)
            gp_sdata_size.append(0)
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
                ch.dtype_fmt = dtype(f"{sig.shape[1:]}{sig.dtype}")

                gp_channels.append(ch)

                offset += byte_size

                gp_sdata.append(None)
                gp_sdata_size.append(0)
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
                        size=data_size,
                        count=len(data),
                        offsets=offsets,
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
                    raw_size=size,
                    size=size,
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
        gp_sdata_size = gp.signal_data_size
        gp_channels = gp.channels
        gp_dep = gp.channel_dependencies

        name = signal.name
        names = signal.samples.dtype.names

        field_name = field_names.get_unique_name(name)

        # first we add the structure channel

        if signal.attachment:
            at_data, at_name = signal.attachment
            attachment_addr = self.attach(at_data, at_name, mime="application/x-dbc")
            attachment = self._attachments_map[attachment_addr]
        else:
            attachment_addr = 0
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

        if attachment_addr:
            kwargs["attachment_addr"] = attachment_addr

        source_bus = signal.source and signal.source.source_type == v4c.SOURCE_BUS

        if source_bus:
            kwargs["flags"] = v4c.FLAG_CN_BUS_EVENT
            flags_ = v4c.FLAG_CN_BUS_EVENT
            grp.channel_group.flags |= v4c.FLAG_CG_BUS_EVENT
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

        if source_bus:
            grp.channel_group.acq_source = SourceInformation.from_common_source(signal.source)

        if source_bus and signal.source.bus_type == v4c.BUS_TYPE_CAN:
            grp.channel_group.path_separator = 46
            grp.CAN_logging = True
            grp.channel_group.acq_name = "CAN"

            can_ids = unique(signal.samples[f"{name}.BusChannel"])

            if name in ('CAN_DataFrame', 'CAN_RemoteFrame'):

                if len(can_ids) == 1:
                    can_id = f"CAN{int(can_ids[0])}"

                    message_ids = set(unique(signal.samples[f"{name}.ID"]))

                    if can_id not in self.can_logging_db:
                        self.can_logging_db[can_id] = {}
                    for message_id in message_ids:
                        self.can_logging_db[can_id][message_id] = dg_cntr
                else:
                    for can_id in can_ids:
                        idx = argwhere(
                            signal.samples[f"{name}.BusChannel"] == can_id
                        ).ravel()
                        message_ids = set(unique(signal.samples[f"{name}.ID"][idx]))
                        can_id = f"CAN{can_id}"
                        if can_id not in self.can_logging_db:
                            self.can_logging_db[can_id] = {}
                        for message_id in message_ids:
                            self.can_logging_db[can_id][message_id] = dg_cntr

        # source for channel
        source = signal.source
        if source:
            if source in si_map:
                ch.source = si_map[source]
            else:
                new_source = SourceInformation(
                    source_type=source.source_type, bus_type=source.bus_type,
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
        gp_sdata_size.append(0)
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
                ch.dtype_fmt = dtype(f"{samples.shape[1:]}{samples.dtype}")

                entry = (dg_cntr, ch_cntr)
                gp_channels.append(ch)
                dep_list.append(entry)

                offset += byte_size

                gp_sdata.append(None)
                gp_sdata_size.append(0)
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
                    gp_sdata_size.append(0)
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
        gp_sdata_size = gp.signal_data_size
        gp_channels = gp.channels
        gp_dep = gp.channel_dependencies

        name = signal.name
        names = signal.samples.dtype.names

        field_name = field_names.get_unique_name(name)

        # first we add the structure channel

        if signal.attachment:
            at_data, at_name = signal.attachment
            attachment_addr = self.attach(at_data, at_name, mime="application/x-dbc")
            attachment = self._attachments_map[attachment_addr]
        else:
            attachment_addr = 0
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

        if attachment_addr:
            kwargs["attachment_addr"] = attachment_addr

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
            grp.channel_group.acq_source = SourceInformation.from_common_source(signal.source)

        if source_bus and signal.source.bus_type == v4c.BUS_TYPE_CAN:
            grp.channel_group.path_separator = 46
            grp.CAN_logging = True
            grp.channel_group.acq_name = "CAN"

            can_ids = unique(signal.samples[f"{name}.BusChannel"])

            if name in ('CAN_DataFrame', 'CAN_RemoteFrame'):

                if len(can_ids) == 1:
                    can_id = f"CAN{int(can_ids[0])}"

                    message_ids = set(unique(signal.samples[f"{name}.ID"]))

                    if can_id not in self.can_logging_db:
                        self.can_logging_db[can_id] = {}
                    for message_id in message_ids:
                        self.can_logging_db[can_id][message_id] = dg_cntr
                else:
                    for can_id in can_ids:
                        idx = argwhere(
                            signal.samples[f"{name}.BusChannel"] == can_id
                        ).ravel()
                        message_ids = set(unique(signal.samples[f"{name}.ID"][idx]))
                        can_id = f"CAN{can_id}"
                        if can_id not in self.can_logging_db:
                            self.can_logging_db[can_id] = {}
                        for message_id in message_ids:
                            self.can_logging_db[can_id][message_id] = dg_cntr

        # source for channel
        source = signal.source
        if source:
            if source in si_map:
                ch.source = si_map[source]
            else:
                new_source = SourceInformation(
                    source_type=source.source_type, bus_type=source.bus_type,
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
        gp_sdata_size.append(0)
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
                ch.dtype_fmt = dtype(f"{samples.shape[1:]}{samples.dtype}")

                entry = (dg_cntr, ch_cntr)
                gp_channels.append(ch)
                dep_list.append(entry)

                offset += byte_size

                gp_sdata.append(None)
                gp_sdata_size.append(0)
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
                    gp_sdata_size.append(0)
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
        >>> mdf.append([s1, s2, s3], 'created by asammdf v1.1.0')
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

                    vals = signal.tostring()

                    fields.append(frombuffer(vals, dtype="V6"))
                    types.append(("", "V6"))

                else:
                    vals = []
                    for field in ("ms", "min", "hour", "day", "month", "year"):
                        vals.append(signal[field])
                    vals = fromarrays(vals).tostring()

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
                cur_offset = sum(blk.size for blk in gp.signal_data[i])

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
                        size=block_size,
                        count=len(values),
                        offsets=offsets,
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
                        raw_size=raw_size,
                        size=size,
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
                        raw_size=raw_size,
                        size=size,
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
                            raw_size=raw_size,
                            size=size,
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
        >>> mdf.append([s1, s2, s3], 'created by asammdf v1.1.0')
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
                cur_offset = sum(blk.size for blk in gp.signal_data[0])

                offsets = arange(len(signal), dtype=uint64) * (signal.itemsize + 4)

                values = [full(len(signal), signal.itemsize, dtype=uint32), signal]

                types_ = [("", uint32), ("", signal.dtype)]

                values = fromarrays(values, dtype=types_)

                addr = tell()
                block_size = len(values) * values.itemsize
                if block_size:
                    info = SignalDataBlockInfo(
                        address=addr,
                        size=block_size,
                        count=len(values),
                        offsets=offsets,
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
                        raw_size=raw_size,
                        size=size,
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
                            raw_size=raw_size,
                            size=size,
                            param=None,
                        )
                    )

    def attach(
        self,
        data,
        file_name=None,
        comment=None,
        compression=True,
        mime=r"application/octet-stream",
        embedded=True,
    ):
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
        embedded : bool
            attachment is embedded in the file

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
                file_name if file_name else "bin.bin", __version__
            )

            self.file_history.append(fh)

            file_name = file_name or "bin.bin"

            at_block = AttachmentBlock(
                data=data,
                compression=compression,
                embedded=embedded,
                file_name=file_name,
            )
            at_block["creator_index"] = creator_index
            index = v4c.MAX_UINT64 - 1
            while index in self._attachments_map:
                index -= 1
            self.attachments.append(at_block)

            at_block.mime = mime
            at_block.comment = comment

            self._attachments_cache[data] = index
            self._attachments_map[index] = len(self.attachments) - 1

            return index

    def close(self):
        """ if the MDF was created with memory=False and new
        channels have been appended, then this must be called just before the
        object is not used anymore to clean-up the temporary file"""

        if self._tempfile is not None:
            self._tempfile.close()
        if self._file is not None:
            self._file.close()

        self.groups.clear()
        self.header = None
        self.identification = None
        self.file_history.clear()
        self.channels_db.clear()
        self.can_logging_db.clear()
        self.masters_db.clear()
        self.attachments.clear()
        self._attachments_cache.clear()
        self.file_comment = None
        self.events.clear()

        self._attachments_map.clear()
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
        data : (bytes, pathlib.Path)
            tuple of attachment data and path

        """
        if address is None and index is None:
            return b"", Path("")

        if address is not None:
            index = self._attachments_map[address]
        attachment = self.attachments[index]

        current_path = Path.cwd()
        file_path = Path(attachment.file_name or "embedded")
        try:
            os.chdir(self.name.resolve().parent)

            flags = attachment.flags

            # for embedded attachments extrat data and create new files
            if flags & v4c.FLAG_AT_EMBEDDED:
                data = attachment.extract()
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
        except Exception as err:
            os.chdir(current_path)
            message = f'Exception during attachment "{attachment.file_name}" extraction: {err!r}'
            logger.warning(message)
            data = b""
        finally:
            os.chdir(current_path)

        return data, file_path

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
        source=None,
        record_offset=0,
        record_count=None,
    ):
        """Gets channel samples. The raw data group samples are not loaded to
        memory so it is advised to use ``filter`` or ``select`` instead of
        performing several ``get`` calls.

        Channel can be specified in two ways:

        * using the first positional argument *name*

            * if *source* is given this will be first used to validate the
              channel selection
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
        ignore_invalidation_bits : bool
            option to ignore invalidation bits
        source : str
            source name used to select the channel
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
        >>> # validation using source name
        ...
        >>> mdf.get('Sig', source='VN7060')
        <Signal Sig:
                samples=[ 12.  12.  12.  12.  12.]
                timestamps=[0 1 2 3 4]
                unit=""
                info=None
                comment="">

        """

        gp_nr, ch_nr = self._validate_channel_selection(
            name, group, index, source=source
        )

        grp = self.groups[gp_nr]

        if ch_nr >= 0:
            # get the channel object
            channel = grp.channels[ch_nr]
            dependency_list = grp.channel_dependencies[ch_nr]

        else:
            channel = grp.logging_channels[-ch_nr - 1]
            dependency_list = None

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

            if vals.dtype.kind == 'S':
                encoding = 'utf-8'

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
                source = SignalSource(
                    source.name,
                    source.path,
                    source.comment,
                    source.source_type,
                    source.bus_type,
                )
            else:
                cg_source = grp.channel_group.acq_source
                if cg_source:
                    source = SignalSource(
                        cg_source.name,
                        cg_source.path,
                        cg_source.comment,
                        cg_source.source_type,
                        cg_source.bus_type,
                    )
                else:
                    source = None

            if hasattr(channel, "attachment_addr"):
                index = self._attachments_map[channel.attachment_addr]
                attachment = self.extract_attachment(index=index)
            elif channel_type == v4c.CHANNEL_TYPE_SYNC and channel.data_block_addr:
                index = self._attachments_map[channel.data_block_addr]
                attachment = self.extract_attachment(index=index)
            else:
                attachment = ()

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
                    timestamps.append(self.get_master(gp_nr, fragment, one_piece=True,))
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
                    )[0]
                    channel_values[i].append(vals)
                if master_is_required:
                    timestamps.append(self.get_master(gp_nr, fragment,))
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
                vals = concatenate(channel_values, out=out,)
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
                vals, timestamps, name="_", invalidation_bits=invalidation_bits,
            ).interp(t, interpolation_mode=self._integer_interpolation,)

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

        channel_group = grp.channel_group
        samples_size = (
            channel_group.samples_byte_nr + channel_group.invalidation_bytes_nr
        )

        channel_values = []
        timestamps = []
        invalidation_bits = []
        count = 0
        for fragment in data:

            data_bytes, offset, _count, invalidation_bytes = fragment

            cycles = len(data_bytes) // samples_size

            arrays = []
            types = []
            try:
                parent, bit_offset = parents[ch_nr]
            except KeyError:
                parent, bit_offset = None, None

            if parent is not None:
                if grp.record is None:
                    dtypes = grp.types
                    if dtypes.itemsize:
                        record = fromstring(data_bytes, dtype=dtypes)
                    else:
                        record = None

                else:
                    record = grp.record

                vals = record[parent].copy()
            else:
                vals = self._get_not_byte_aligned_data(data_bytes, grp, ch_nr)

            dep = dependency_list[0]
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
                            axis = array([axis for _ in range(cycles_nr)])
                            arrays.append(axis)
                            dtype_pair = (f"axis_{i}", axis.dtype, shape)
                            types.append(dtype_pair)
                    else:
                        for i in range(dims_nr):

                            axis = ca_block.axis_channels[i]
                            shape = (ca_block[f"dim_size_{i}"],)

                            if axis is None:
                                axisname = f"axis_{i}"
                                if cycles:
                                    axis_values = array([arange(shape[0])] * cycles)
                                else:
                                    axis_values = array([], dtype=f"({shape[0]},)f8")

                            else:
                                try:
                                    (ref_dg_nr, ref_ch_nr,) = ca_block.axis_channels[i]
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
                        axis = array([axis for _ in range(cycles_nr)])
                        arrays.append(axis)
                        types.append((f"axis_{i}", axis.dtype, shape))
                else:
                    for i in range(dims_nr):
                        axis = ca_block.axis_channels[i]
                        shape = (ca_block[f"dim_size_{i}"],)

                        if axis is None:
                            axisname = f"axis_{i}"
                            if cycles:
                                axis_values = array([arange(shape[0])] * cycles)
                            else:
                                axis_values = array([], dtype=f"({shape[0]},)f8")

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
                timestamps.append(self.get_master(gp_nr, fragment))
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
                vals, timestamps, name="_", invalidation_bits=invalidation_bits,
            ).interp(t, interpolation_mode=self._integer_interpolation,)

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
                    timestamps.append(self.get_master(gp_nr, fragment))
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
                    vals, timestamps, name="_", invalidation_bits=invalidation_bits,
                ).interp(t, interpolation_mode=self._integer_interpolation,)

                vals, timestamps, invalidation_bits = (
                    vals.samples,
                    vals.timestamps,
                    vals.invalidation_bits,
                )

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
                    elif len(shape_) > 1 and data_type != v4c.DATA_TYPE_BYTEARRAY:
                        vals = self._get_not_byte_aligned_data(data_bytes, grp, ch_nr)
                    elif vals_dtype not in "ui" and (bit_offset or not bit_count == size * 8):
                        vals = self._get_not_byte_aligned_data(data_bytes, grp, ch_nr)
                    else:
                        dtype_ = vals.dtype
                        kind_ = dtype_.kind

                        if data_type in v4c.INT_TYPES:

                            if channel.dtype_fmt.subdtype:
                                channel_dtype = channel.dtype_fmt.subdtype[0]
                            else:
                                channel_dtype = channel.dtype_fmt

                            if channel_dtype.byteorder == '|' and data_type in (v4c.DATA_TYPE_SIGNED_MOTOROLA, v4c.DATA_TYPE_UNSIGNED_MOTOROLA):
                                view = f'>u{vals.itemsize}'
                            else:
                                view = f'{channel_dtype.byteorder}u{vals.itemsize}'

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
                                view = f'{channel_dtype.byteorder}i{vals.itemsize}'
                                vals = vals.view(view)

                        else:
                            if bit_count != size * 8:
                                vals = self._get_not_byte_aligned_data(
                                    data_bytes, grp, ch_nr
                                )
                            else:
                                if kind_ in "ui":
                                    if channel.dtype_fmt.subdtype:
                                        channel_dtype = channel.dtype_fmt.subdtype[0]
                                    else:
                                        channel_dtype = channel.dtype_fmt
                                    vals = vals.view(channel_dtype)

                else:
                    vals = self._get_not_byte_aligned_data(data_bytes, grp, ch_nr)

                if self._single_bit_uint_as_bool and bit_count == 1:
                    vals = array(vals, dtype=bool)
                else:
                    if channel.dtype_fmt.subdtype:
                        channel_dtype = channel.dtype_fmt.subdtype[0]
                    else:
                        channel_dtype = channel.dtype_fmt
                    if vals.dtype != channel_dtype:
                        vals = vals.astype(channel_dtype)

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
                        elif len(shape_) > 1 and data_type != v4c.DATA_TYPE_BYTEARRAY:
                            vals = self._get_not_byte_aligned_data(
                                data_bytes, grp, ch_nr
                            )
                        elif vals_dtype not in "ui" and (bit_offset or not bit_count == size * 8):
                            vals = self._get_not_byte_aligned_data(data_bytes, grp, ch_nr)
                        else:
                            dtype_ = vals.dtype
                            kind_ = dtype_.kind

                            if data_type in v4c.INT_TYPES:

                                if channel.dtype_fmt.subdtype:
                                    channel_dtype = channel.dtype_fmt.subdtype[0]
                                else:
                                    channel_dtype = channel.dtype_fmt

                                if channel_dtype.byteorder == '|' and data_type in (v4c.DATA_TYPE_SIGNED_MOTOROLA, v4c.DATA_TYPE_UNSIGNED_MOTOROLA):
                                    view = f'>u{vals.itemsize}'
                                else:
                                    view = f'{channel_dtype.byteorder}u{vals.itemsize}'
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
                                    view = f'{channel_dtype.byteorder}i{vals.itemsize}'
                                    vals = vals.view(view)

                            else:
                                if bit_count != size * 8:
                                    vals = self._get_not_byte_aligned_data(
                                        data_bytes, grp, ch_nr
                                    )
                                else:
                                    if kind_ in "ui":
                                        if channel.dtype_fmt.subdtype:
                                            channel_dtype = channel.dtype_fmt.subdtype[0]
                                        else:
                                            channel_dtype = channel.dtype_fmt
                                        vals = vals.view(channel_dtype)

                    else:
                        vals = self._get_not_byte_aligned_data(data_bytes, grp, ch_nr)

                    if bit_count == 1 and self._single_bit_uint_as_bool:
                        vals = array(vals, dtype=bool)
                    else:

                        if channel.dtype_fmt.subdtype:
                            channel_dtype = channel.dtype_fmt.subdtype[0]
                        else:
                            channel_dtype = channel.dtype_fmt
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
                    vals, timestamps, name="_", invalidation_bits=invalidation_bits,
                ).interp(t, interpolation_mode=self._integer_interpolation,)

                vals, timestamps, invalidation_bits = (
                    vals.samples,
                    vals.timestamps,
                    vals.invalidation_bits,
                )

        if channel_type == v4c.CHANNEL_TYPE_VLSD:
            count_ = len(vals)

            signal_data, with_bounds = self._load_signal_data(
                group=grp, index=ch_nr, offset=record_start, count=count_,
            )

            if signal_data:
                if data_type in (v4c.DATA_TYPE_BYTEARRAY, v4c.DATA_TYPE_UNSIGNED_INTEL, v4c.DATA_TYPE_UNSIGNED_MOTOROLA):
                    vals = extract(signal_data, 1)
                else:
                    vals = extract(signal_data, 0)

                if not with_bounds:
                    vals = vals[record_start : record_start + count_]

                if data_type not in (v4c.DATA_TYPE_BYTEARRAY, v4c.DATA_TYPE_UNSIGNED_INTEL, v4c.DATA_TYPE_UNSIGNED_MOTOROLA):

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

        if ch_nr >= 0:
            channel = group.channels[ch_nr]
        else:
            channel = group.logging_channels[-ch_nr - 1]

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
        else:
            extra_bytes = 4 - (byte_size % 4)

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
                    [vals, zeros(len(vals), dtype=f"<({extra_bytes},)u1"),]
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
        self,
        index=None,
        channels=None,
        skip_master=True,
        minimal=True,
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

                if group.CAN_logging:
                    found = True
                    where = (
                        self.whereis("CAN_DataFrame")
                        + self.whereis("CAN_ErrorFrame")
                        + self.whereis("CAN_RemoteFrame")
                    )

                    for dg_cntr, ch_cntr in where:
                        if dg_cntr == gp_index:
                            break
                    else:
                        found = False
                        group.CAN_logging = False

                    if found:
                        channel = channels[ch_cntr]

                        frame_bytes = range(
                            channel.byte_offset,
                            channel.byte_offset + channel.bit_count // 8,
                        )

                        for i, channel in enumerate(channels):
                            if channel.byte_offset in frame_bytes:
                                included_channels.remove(i)

                        included_channels.add(ch_cntr)

                        if group.CAN_database:
                            dbc_addr = group.dbc_addr
                            message_id = group.message_id
                            for m_ in message_id:
                                try:
                                    can_msg = self._dbc_cache[dbc_addr].frameById(m_)
                                except AttributeError:
                                    can_msg = self._dbc_cache[dbc_addr].frame_by_id(
                                        canmatrix.ArbitrationId(m_)
                                    )

                                for i, _ in enumerate(can_msg.signals, 1):
                                    included_channels.add(-i)

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
                            "or they must match the first 3 argumens of the get "
                            "method"
                        )
                    else:
                        group, idx = self._validate_channel_selection(*item)
                        if group not in gps:
                            gps[group] = {idx}
                        else:
                            gps[group].add(idx)
                else:
                    name = item
                    group, idx = self._validate_channel_selection(name)
                    if group not in gps:
                        gps[group] = {idx}
                    else:
                        gps[group].add(idx)

            result = {}
            for gp_index, channels in gps.items():
                master = self.virtual_groups_map[gp_index]
                group = self.groups[gp_index]

                if minimal:

                    channel_dependencies = [
                        group.channel_dependencies[ch_nr]
                        for ch_nr in channels
                    ]

                    for dependencies in channel_dependencies:
                        if dependencies is None:
                            continue

                        if all(
                            not isinstance(dep, ChannelArrayBlock) for dep in dependencies
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

                if master not in result:
                    result[master] = {}
                    result[master][master] = [self.masters_db[master]]

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
                count = 8 * 1024 * 1024 // record_size or 1
            else:
                count = 128 * 1024 * 1024 // record_size or 1

        data_streams = []
        for idx, group_index in enumerate(groups):
            grp = self.groups[group_index]
            grp.read_split_count = count
            data_streams.append(
                self._load_data(
                    grp, record_offset=record_offset, record_count=record_count,
                )
            )
            if group_index == index:
                master_index = idx

        encodings = {group_index: [None,] for groups_index in groups}

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
                                            decode(sig.samples, "utf-16-be"), "latin-1",
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
        """ returns master channel samples for given group

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
                "the argument raster is depreacted since version 5.13.0 "
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
                offset = offset // record_size
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
                offset = offset // record_size
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

    def get_can_signal(
        self, name, database=None, db=None, ignore_invalidation_bits=False,
        data=None,
    ):
        """ get CAN message signal. You can specify an external CAN database (
        *database* argument) or canmatrix databse object that has already been
        loaded from a file (*db* argument).

        The signal name can be specified in the following ways

        * ``CAN<ID>.<MESSAGE_NAME>.<SIGNAL_NAME>`` - the `ID` value starts from 1
          and must match the ID found in the measurement (the source CAN bus ID)
          Example: CAN1.Wheels.FL_WheelSpeed

        * ``CAN<ID>.CAN_DataFrame_<MESSAGE_ID>.<SIGNAL_NAME>`` - the `ID` value
          starts from 1 and the `MESSAGE_ID` is the decimal message ID as found
          in the database. Example: CAN1.CAN_DataFrame_218.FL_WheelSpeed

        * ``<MESSAGE_NAME>.SIGNAL_NAME`` - in this case the first occurence of
          the message name and signal are returned (the same message could be
          found on muplit CAN buses; for example on CAN1 and CAN3)
          Example: Wheels.FL_WheelSpeed

        * ``CAN_DataFrame_<MESSAGE_ID>.<SIGNAL_NAME>`` - in this case the first
          occurence of the message name and signal are returned (the same
          message could be found on muplit CAN buses; for example on CAN1 and
          CAN3). Example: CAN_DataFrame_218.FL_WheelSpeed

        * ``<SIGNAL_NAME>`` - in this case the first occurence of the signal
          name is returned ( the same signal anme coudl be found in multiple
          messages and on multiple CAN buses). Example: FL_WheelSpeed


        Parameters
        ----------
        name : str
            signal name
        database : str
            path of external CAN database file (.dbc or .arxml); default *None*
        db : canmatrix.database
            canmatrix CAN database object; default *None*
        ignore_invalidation_bits : bool
            option to ignore invalidation bits

        Returns
        -------
        sig : Signal
            Signal object with the physical values

        """

        if database is None and db is None:
            return self.get(name)

        if db is None:

            if not str(database).lower().endswith(("dbc", "arxml")):
                message = f'Expected .dbc or .arxml file as CAN channel attachment but got "{database}"'
                logger.exception(message)
                raise MdfException(message)
            else:
                db_string = Path(database).read_bytes()
                md5_sum = md5(db_string).digest()

                if md5_sum in self._external_dbc_cache:
                    db = self._external_dbc_cache[md5_sum]
                else:
                    db = load_can_database(database, db_string)
                    if db is None:
                        raise MdfException("failed to load database")

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
                can_id = f'CAN{can_id.group("id")}'

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
            for _can_id, messages in self.can_logging_db.items():

                if is_j1939:
                    test_ids = [
                        canmatrix.ArbitrationId(id_, extended=True).pgn
                        for id_ in self.can_logging_db[_can_id]
                    ]

                    id_ = message.arbitration_id.pgn

                else:
                    id_ = message.arbitration_id.id
                    test_ids = self.can_logging_db[_can_id]

                if id_ in test_ids:
                    if is_j1939:
                        for id__, idx in self.can_logging_db[_can_id].items():
                            if canmatrix.ArbitrationId(id__, extended=True).pgn == id_:
                                index = idx
                                break
                    else:
                        index = self.can_logging_db[_can_id][message.arbitration_id.id]

                if index is not None:
                    break
            else:
                raise MdfException(
                    f'Message "{message.name}" (ID={hex(message.arbitration_id.id)}) not found in the measurement'
                )
        else:
            if can_id in self.can_logging_db:
                if is_j1939:
                    test_ids = [
                        canmatrix.ArbitrationId(id_, extended=True).pgn
                        for id_ in self.can_logging_db[can_id]
                    ]
                    id_ = message.arbitration_id.pgn

                else:
                    id_ = message.arbitration_id.id
                    test_ids = self.can_logging_db[can_id]

                if id_ in test_ids:
                    if is_j1939:
                        for id__, idx in self.can_logging_db[can_id].items():
                            if canmatrix.ArbitrationId(id__, extended=True).pgn == id_:
                                index = idx
                                break
                    else:
                        index = self.can_logging_db[can_id][message.arbitration_id.id]
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
        can_ids.samples = can_ids.samples & 0x1fffffff
        payload = self.get(
            "CAN_DataFrame.DataBytes",
            group=index,
            samples_only=True,
            ignore_invalidation_bits=ignore_invalidation_bits,
            data=data,
        )[0]

        if is_j1939:
            ps = (can_ids.samples >> 8) & 0xFF
            pf = (can_ids.samples >> 16) & 0xFF
            _pgn = pf << 8
            _pgn = where(pf >= 240, _pgn + ps, _pgn,)

            idx = nonzero(_pgn == message.arbitration_id.pgn)[0]
        else:
            idx = nonzero(can_ids.samples == message.arbitration_id.id)[0]

        vals = payload[idx]
        t = can_ids.timestamps[idx].copy()

        if can_ids.invalidation_bits is not None:
            invalidation_bits = can_ids.invalidation_bits[idx]
        else:
            invalidation_bits = None

        vals = extract_can_signal(signal, vals)

        comment = signal.comment or ""

        if ignore_invalidation_bits:

            sig = Signal(
                samples=vals,
                timestamps=t,
                name=name,
                unit=signal.unit or "",
                comment=comment,
                invalidation_bits=invalidation_bits,
            )
            return sig

        else:

            if invalidation_bits is not None:
                vals = vals[nonzero(~invalidation_bits)[0]]
                t = t[nonzero(~invalidation_bits)[0]]

            return Signal(
                samples=vals,
                timestamps=t,
                name=name,
                unit=signal.unit or "",
                comment=comment,
            )

    def info(self):
        """get MDF information as a dict

        Examples
        --------
        >>> mdf = MDF4('test.mdf')
        >>> mdf.info()


        """
        info = {}
        info["version"] = (
            self.identification["version_str"].decode("utf-8").strip(" \n\t\0")
        )
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
        is overwritten, otherwise the file name is appened with '.<cntr>', were
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
        output_file : pathlib.Path
            path to saved file

        """

        if is_file_like(dst):
            dst_ = dst
            file_like = True
            dst = Path("__file_like.mf4")
            dst_.seek(0)
        else:
            file_like = False
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

            # attachments
            at_map = {}
            if self.attachments:
                for at_block in self.attachments:
                    address = at_block.to_blocks(address, blocks, defined_texts)

                for i in range(len(self.attachments) - 1):
                    at_block = self.attachments[i]
                    at_block.next_at_addr = self.attachments[i + 1].address
                self.attachments[-1].next_at_addr = 0

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
                        if channel.data_block_addr:
                            idx = self._attachments_map[channel.data_block_addr]
                            channel.data_block_addr = self.attachments[idx].address
                    else:
                        sdata, with_bounds = self._load_signal_data(group=gp, index=j)
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

                for channel in gp.logging_channels:
                    address = channel.to_blocks(
                        address, blocks, defined_texts, cc_map, si_map
                    )

                group_channels = list(chain(gp.channels, gp.logging_channels))
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
                    for i in range(event.attachment_nr):
                        key = f"attachment_{i}_addr"
                        addr = event[key]
                        event[key] = at_map[addr]

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

        if dst == self.name:
            self.close()
            Path.unlink(self.name)
            Path.rename(destination, self.name)

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

        if ch_nr >= 0:
            channel = grp.channels[ch_nr]
        else:
            channel = grp.logging_channels[-ch_nr - 1]

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

        flags = self.identification["unfinalized_standard_flags"]

        shim = FinalizationShim(self._file, flags)
        shim.load_blocks()

        shim.finalize()

        # In-memory finalization performed, inject as a shim between the original file and asammdf.
        self._file_orig = self._file
        self._file = shim

        self.identification.file_identification = b"MDF     "
        self.identification.unfinalized_standard_flags = 0
        self.identification.unfinalized_custom_flags = 0

    def _sort(self):
        if self._file is None:
            return
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

            rem = b''
            for info in group.data_blocks:
                address, size, block_size, block_type, param = (
                    info.address,
                    info.raw_size,
                    info.size,
                    info.block_type,
                    info.param,
                )

                if block_type != v4c.DT_BLOCK:
                    partial_records = {id_: [] for _, id_ in groups}
                    new_data = read(block_size)

                    if block_type == v4c.DZ_BLOCK_DEFLATE:
                        new_data = decompress(new_data, 0, size)
                    elif block_type == v4c.DZ_BLOCK_TRANSPOSED:
                        new_data = decompress(new_data, 0, size)
                        cols = param
                        lines = size // cols

                        nd = fromstring(new_data[: lines * cols], dtype=uint8)
                        nd = nd.reshape((cols, lines))
                        new_data = nd.T.tostring() + new_data[lines * cols :]

                    new_data = rem + new_data

                    try:
                        rem = sort_data_block(
                            new_data, partial_records, cg_size, record_id_nr, _unpack_stuct
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

                            address = tell()

                            size = write(b"".join(new_data))

                            if dg_cntr is not None:
                                offsets, size = get_vlsd_offsets(new_data)

                                if size:
                                    info = SignalDataBlockInfo(
                                        address=address,
                                        size=size,
                                        count=len(offsets),
                                        offsets=offsets,
                                    )
                                    self.groups[dg_cntr].signal_data[ch_cntr].append(info)

                            else:
                                if size:
                                    block_info = DataBlockInfo(
                                        address=address,
                                        block_type=v4c.DT_BLOCK,
                                        raw_size=size,
                                        size=size,
                                        param=0,
                                    )
                                    final_records[rec_id].append(block_info)
                                    size = 0
                else:

                    seek(address)
                    limit = 32 * 1024 * 1024
                    while block_size:
                        if block_size > limit:
                            block_size -= limit
                            new_data = rem + read(limit)
                        else:
                            new_data = rem + read(block_size)
                            block_size = 0
                        partial_records = {id_: [] for _, id_ in groups}

                        rem = sort_data_block(
                            new_data, partial_records, cg_size, record_id_nr, _unpack_stuct
                        )

                        for rec_id, new_data in partial_records.items():

                            channel_group = cg_map[rec_id]

                            if channel_group.address in self._cn_data_map:
                                dg_cntr, ch_cntr = self._cn_data_map[channel_group.address]
                            else:
                                dg_cntr, ch_cntr = None, None

                            if new_data:

                                if dg_cntr is not None:
                                    address = tell()
                                    size = write(b"".join(new_data))

                                    offsets, size = get_vlsd_offsets(new_data)

                                    if size:
                                        info = SignalDataBlockInfo(
                                            address=address,
                                            size=size,
                                            count=len(offsets),
                                            offsets=offsets,
                                        )
                                        self.groups[dg_cntr].signal_data[ch_cntr].append(info)

                                else:
                                    if size:
#                                        block_info = DataBlockInfo(
#                                            address=address,
#                                            block_type=v4c.DT_BLOCK,
#                                            raw_size=size,
#                                            size=size,
#                                            param=0,
#                                        )
#                                        final_records[rec_id].append(block_info)
#                                        size = 0

                                        address = tell()

                                        new_data = b"".join(new_data)

                                        raw_size = len(new_data)
                                        new_data = lz_compress(new_data)
                                        size = len(new_data)
                                        self._tempfile.write(new_data)

                                        block_info = InvalidationBlockInfo(
                                            address=address,
                                            block_type=v4c.DZ_BLOCK_LZ,
                                            raw_size=raw_size,
                                            size=size,
                                            param=None,
                                        )

                                        final_records[rec_id].append(block_info)
                                        size = 0

            for idx, rec_id in groups:
                group = self.groups[idx]

                group.data_location = v4c.LOCATION_TEMPORARY_FILE
                group.set_blocks_info(final_records[rec_id])
                group.sorted = True

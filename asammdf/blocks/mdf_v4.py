"""
ASAM MDF version 4 file format module
"""

import logging
import xml.etree.ElementTree as ET
import os
from copy import deepcopy
from functools import reduce
from hashlib import md5
from itertools import chain
from math import ceil
from struct import unpack, unpack_from
from tempfile import TemporaryFile
from zlib import decompress
from pathlib import Path
import mmap

from numpy import (
    arange,
    array,
    array_equal,
    concatenate,
    dtype,
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
    uint64,
    union1d,
    unpackbits,
    zeros,
    uint32,
    fliplr,
    searchsorted,
    full,
)

from numpy.core.records import fromarrays, fromstring
from canmatrix.formats import loads
from pandas import DataFrame

from . import v4_constants as v4c
from ..signal import Signal
from .conversion_utils import conversion_transfer
from .utils import (
    UINT8_u,
    UINT16_u,
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
    SourceInformation,
    TextBlock,
)
from ..version import __version__


MASTER_CHANNELS = (v4c.CHANNEL_TYPE_MASTER, v4c.CHANNEL_TYPE_VIRTUAL_MASTER)
COMMON_SIZE = v4c.COMMON_SIZE
COMMON_u = v4c.COMMON_u
COMMON_uf = v4c.COMMON_uf


logger = logging.getLogger("asammdf")

__all__ = ["MDF4"]


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
        mdf file version ('4.00', '4.10', '4.11'); default '4.10'
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
        self._master_channel_cache = {}
        self._master_channel_metadata = {}
        self._invalidation_cache = {}
        self._si_map = {}
        self._file_si_map = {}
        self._cc_map = {}
        self._file_cc_map = {}
        self._cg_map = {}
        self._dbc_cache = {}

        self._tempfile = TemporaryFile()
        self._file = None

        self._read_fragment_size = 0 * 2 ** 20
        self._write_fragment_size = 4 * 2 ** 20
        self._use_display_names = kwargs.get("use_display_names", False)
        self._single_bit_uint_as_bool = False
        self._integer_interpolation = 0

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
                self.name = Path(name)
                with open(self.name, "rb") as x:
                    self._file = mmap.mmap(x.fileno(), 0, access=mmap.ACCESS_READ)
                    self._from_filelike = False
                    self._read(mapped=True)

                    self._file.close()

                self._file = open(self.name, "rb")

        else:
            self._from_filelike = False
            version = validate_version_argument(version)
            self.header = HeaderBlock()
            self.identification = FileIdentificationBlock(version=version)
            self.version = version
            self.name = Path("new.mf4")

    def _check_finalised(self):
        flags = self.identification["unfinalized_standard_flags"]
        if flags & 1:
            message = (
                f"Unfinalised file {self.name}:"
                "Update of cycle counters for CG/CA blocks required"
            )

            logger.warning(message)
        elif flags & 1 << 1:
            message = f"Unfinalised file {self.name}: Update of cycle counters for SR blocks required"

            logger.warning(message)
        elif flags & 1 << 2:
            message = f"Unfinalised file {self.name}: Update of length for last DT block required"

            logger.warning(message)
        elif flags & 1 << 3:
            message = f"Unfinalised file {self.name}: Update of length for last RD block required"

            logger.warning(message)
        elif flags & 1 << 4:
            message = (
                f"Unfinalised file {self.name}:"
                "Update of last DL block in each chained list"
                "of DL blocks required"
            )

            logger.warning(message)
        elif flags & 1 << 5:
            message = (
                f"Unfinalised file {self.name}:"
                "Update of cg_data_bytes and cg_inval_bytes "
                "in VLSD CG block required"
            )

            logger.warning(message)
        elif flags & 1 << 6:
            message = (
                f"Unfinalised file {self.name}:"
                "Update of offset values for VLSD channel required "
                "in case a VLSD CG block is used"
            )

            logger.warning(message)

    def _read(self, mapped=False):

        stream = self._file
        dg_cntr = 0

        cg_count, _ = count_channel_groups(stream)
        if self._callback:
            self._callback(0, cg_count)
        current_cg_index = 0

        self.identification = FileIdentificationBlock(stream=stream, mapped=mapped)
        version = self.identification["version_str"]
        self.version = version.decode("utf-8").strip(" \n\t\0")

        if self.version >= "4.10":
            self._check_finalised()

        self.header = HeaderBlock(address=0x40, stream=stream, mapped=mapped)

        # read file history
        fh_addr = self.header["file_history_addr"]
        while fh_addr:
            history_block = FileHistory(address=fh_addr, stream=stream, mapped=mapped)
            self.file_history.append(history_block)
            fh_addr = history_block.next_fh_addr

        # read attachments
        at_addr = self.header["first_attachment_addr"]
        index = 0
        while at_addr:
            at_block = AttachmentBlock(address=at_addr, stream=stream, mapped=mapped)
            self._attachments_map[at_addr] = index
            self.attachments.append(at_block)
            at_addr = at_block.next_at_addr
            index += 1

        # go to first date group and read each data group sequentially
        dg_addr = self.header.first_dg_addr

        while dg_addr:
            new_groups = []
            group = DataGroup(address=dg_addr, stream=stream, mapped=mapped)
            record_id_nr = group.record_id_len

            # go to first channel group of the current data group
            cg_addr = group.first_cg_addr

            cg_nr = 0

            cg_size = {}

            while cg_addr:
                cg_nr += 1

                grp = Group(group.copy())

                # read each channel group sequentially
                block = ChannelGroup(address=cg_addr, stream=stream, mapped=mapped)
                self._cg_map[cg_addr] = dg_cntr
                channel_group = grp.channel_group = block

                grp.record_size = cg_size

                if channel_group.flags & v4c.FLAG_CG_VLSD:
                    # VLDS flag
                    record_id = channel_group.record_id
                    cg_size[record_id] = 0
                elif channel_group.flags & v4c.FLAG_CG_BUS_EVENT:
                    bus_type = channel_group.acq_source.bus_type
                    if bus_type == v4c.BUS_TYPE_CAN:
                        grp.CAN_logging = True
                        message_name = channel_group.acq_name

                        comment = channel_group.acq_source.comment
                        if comment:
                            comment_xml = ET.fromstring(comment)
                            common_properties = comment_xml.find(".//common_properties")
                            for e in common_properties:
                                name = e.get("name")
                                if name == "ChannelNo":
                                    grp.CAN_id = f"CAN{e.text}"
                                    break
                        if grp.CAN_id is None:
                            grp.CAN_logging = False
                        else:

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
                                        common_properties = comment_xml.find(
                                            ".//common_properties"
                                        )
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
                                            grp.message_id = message_id

                                    else:
                                        message = f"Invalid bus logging channel group metadata: {comment}"
                                        logger.warning(message)
                                else:
                                    message = (
                                        f"Unable to get CAN message information "
                                        f"since channel group @{hex(channel_group.address)} has no metadata"
                                    )
                                    logger.warning(message)
                    else:
                        # only CAN bus logging is supported
                        pass
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
                neg_ch_cntr = -1

                # Read channels by walking recursively in the channel group
                # starting from the first channel
                self._read_channels(ch_addr, grp, stream, dg_cntr, ch_cntr, neg_ch_cntr, mapped=mapped)

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

            info = self._get_data_blocks_info(
                address=address, stream=stream, block_type=b"##DT", mapped=mapped
            )

            for grp in new_groups:
                grp.data_location = v4c.LOCATION_ORIGINAL_FILE
                grp.set_blocks_info(info)

            self.groups.extend(new_groups)

            dg_addr = group.next_dg_addr

        # all channels have been loaded so now we can link the
        # channel dependencies and load the signal data for VLSD channels
        for grp in self.groups:
            for dep_list in grp.channel_dependencies:
                if not dep_list:
                    continue

                for dep in dep_list:
                    if isinstance(dep, ChannelArrayBlock):
                        conditions = (
                            dep.ca_type == v4c.CA_TYPE_LOOKUP,
                            dep.links_nr == 4 * dep.dims + 1,
                        )
                        if not all(conditions):
                            continue

                        for i in range(dep.dims):
                            ch_addr = dep[f"scale_axis_{i}_ch_addr"]
                            ref_channel = self._ch_map[ch_addr]
                            dep.referenced_channels.append(ref_channel)
                    else:
                        break

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
                    self.can_logging_db[group.CAN_id][message_id] = i
            else:
                continue

            if group.raw_can:
                can_ids = self.get(
                    "CAN_DataFrame.ID", group=i, ignore_invalidation_bits=True
                )
                all_can_ids = sorted(set(can_ids.samples))
                payload = self.get(
                    "CAN_DataFrame.DataBytes",
                    group=i,
                    samples_only=True,
                    ignore_invalidation_bits=True,
                )[0]

                _sig = self.get("CAN_DataFrame", group=i, ignore_invalidation_bits=True)

                attachment = _sig.attachment
                if attachment and attachment[0] and attachment[1].name.lower().endswith(("dbc", "arxml")):
                    attachment, at_name = attachment

                    import_type = "dbc" if at_name.name.lower().endswith("dbc") else "arxml"
                    db = loads(
                        attachment.decode("utf-8"), importType=import_type, key="db"
                    )["db"]

                    board_units = set(bu.name for bu in db.boardUnits)

                    cg_source = group.channel_group.acq_source

                    all_message_info_extracted = True
                    for message_id in all_can_ids:
                        self.can_logging_db[group.CAN_id][message_id] = i
                        sigs = []
                        can_msg = db.frameById(message_id)

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

                            for signal in sorted(can_msg.signals, key=lambda x: x.name):

                                sig_vals = self.get_can_signal(
                                    f"CAN{group.CAN_id}.{can_msg.name}.{signal.name}",
                                    db=db,
                                    ignore_invalidation_bits=True,
                                ).samples

                                conversion = ChannelConversion(
                                    a=float(signal.factor),
                                    b=float(signal.offset),
                                    conversion_type=v4c.CONVERSION_TYPE_LIN,
                                )
                                conversion.unit = signal.unit or ""
                                sigs.append(
                                    Signal(
                                        sig_vals,
                                        t,
                                        name=signal.name,
                                        conversion=conversion,
                                        source=source,
                                        unit=signal.unit,
                                        raw=True,
                                        invalidation_bits=invalidation_bits,
                                    )
                                )

                            processed_can.append(
                                [sigs, message_id, message_name, cg_source, group.CAN_id]
                            )
                        else:
                            all_message_info_extracted = False

                    if all_message_info_extracted:
                        raw_can.append(i)

        # delete the groups that contain raw CAN bus logging and also
        # delete the channel entries from the channels_db. Update data group
        # index for the remaining channel entries. Append new data groups
        if processed_can:
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

            for sigs, message_id, message_name, cg_source, can_id in processed_can:
                self.append(
                    sigs, "Extracted from raw CAN bus logging", common_timebase=True
                )
                group = self.groups[-1]
                group.CAN_database = message_name != ""
                group.CAN_logging = True
                group.CAN_id = can_id
                if message_id > 0:
                    if message_id > 0x80000000:
                        message_id -= 0x80000000
                        group.extended_id = True
                    else:
                        group.extended_id = False
                    group.message_name = message_name
                    group.message_id = message_id
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
                self.can_logging_db[group.CAN_id][group.message_id] = i

        # read events
        addr = self.header.first_event_addr
        ev_map = {}
        event_index = 0
        while addr:
            event = EventBlock(address=addr, stream=stream, mapped=mapped)
            event.update_references(self._ch_map, self._cg_map)
            self.events.append(event)
            ev_map[addr] = event_index
            event_index += 1

            addr = event.next_ev_addr

        for event in self.events:
            addr = event.parent_ev_addr
            if addr:
                event.parent = ev_map[addr]

            addr = event.range_start_ev_addr
            if addr:
                event.range_start = ev_map[addr]

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
        channel_composition=False,
        mapped=False,
    ):

        channels = grp.channels

        dependencies = grp.channel_dependencies

        composition = []
        composition_dtype = []
        while ch_addr:
            # read channel block and create channel object

            channel = Channel(
                address=ch_addr,
                stream=stream,
                cc_map=self._cc_map,
                si_map=self._si_map,
                at_map=self._attachments_map,
                use_display_names=self._use_display_names,
                mapped=mapped,
            )
            value = channel
            display_name = channel.display_name
            name = channel.name

            entry = (dg_cntr, ch_cntr)
            self._ch_map[ch_addr] = entry

            channels.append(value)
            if channel_composition:
                composition.append(entry)

            self.channels_db.add(display_name, entry)
            self.channels_db.add(name, entry)

            # signal data
            address = channel.data_block_addr
            grp.signal_data.append(address)

            if channel.channel_type in MASTER_CHANNELS:
                self.masters_db[dg_cntr] = ch_cntr

            ch_cntr += 1

            component_addr = channel.component_addr

            if component_addr:
                # check if it is a CABLOCK or CNBLOCK
                stream.seek(component_addr)
                blk_id = stream.read(4)
                if blk_id == b"##CN":
                    index = ch_cntr - 1
                    dependencies.append(None)
                    ch_cntr, neg_ch_cntr, ret_composition, ret_composition_dtype = self._read_channels(
                        component_addr, grp, stream, dg_cntr, ch_cntr, neg_ch_cntr, True, mapped=mapped
                    )
                    dependencies[index] = ret_composition

                    channel.dtype_fmt = ret_composition_dtype
                    composition_dtype.append((channel.name, channel.dtype_fmt))

                    if grp.CAN_id is not None and grp.message_id is not None:
                        try:
                            addr = channel.attachment_addr
                        except AttributeError:
                            raise
                            addr = 0
                        if addr:
                            attachment_addr = self._attachments_map[addr]
                            if attachment_addr not in self._dbc_cache:
                                attachment, at_name = self.extract_attachment(
                                    index=attachment_addr
                                )
                                if (
                                    not at_name.name.lower().endswith(("dbc", "arxml"))
                                    or not attachment
                                ):
                                    message = f'Expected .dbc or .arxml file as CAN channel attachment but got "{at_name}"'
                                    logger.warning(message)
                                    grp.CAN_database = False
                                else:
                                    import_type = (
                                        "dbc"
                                        if at_name.name.lower().endswith("dbc")
                                        else "arxml"
                                    )
                                    try:
                                        attachment_string = attachment.decode("utf-8")
                                        self._dbc_cache[attachment_addr] = loads(
                                            attachment_string,
                                            importType=import_type,
                                            key="db",
                                        )["db"]
                                        grp.CAN_database = True
                                    except UnicodeDecodeError:
                                        try:
                                            from cchardet import detect

                                            encoding = detect(attachment)["encoding"]
                                            attachment_string = attachment.decode(
                                                encoding
                                            )
                                            self._dbc_cache[attachment_addr] = loads(
                                                attachment_string,
                                                importType=import_type,
                                                key="db",
                                                encoding=encoding,
                                            )["db"]
                                            grp.CAN_database = True
                                        except ImportError:
                                            message = (
                                                "Unicode exception occured while processing the database "
                                                f'attachment "{at_name}" and "cChardet" package is '
                                                'not installed. Mdf version 4 expects "utf-8" '
                                                "strings and this package may detect if a different"
                                                " encoding was used"
                                            )
                                            logger.warning(message)
                                            grp.CAN_database = False
                            else:
                                grp.CAN_database = True
                        else:
                            grp.CAN_database = False

                        if grp.CAN_database:

                            # here we make available multiple ways to refer to
                            # CAN signals by using fake negative indexes for
                            # the channel entries in the channels_db

                            grp.dbc_addr = attachment_addr

                            message_id = grp.message_id
                            message_name = grp.message_name
                            can_id = grp.CAN_id

                            can_msg = self._dbc_cache[attachment_addr].frameById(
                                message_id
                            )
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
                                little_endian = (
                                    True if signal.is_little_endian else False
                                )
                                signed = signal.is_signed
                                s_type = info_to_datatype_v4(signed, little_endian)
                                bit_offset = signal.startBit % 8
                                byte_offset = signal.startBit // 8
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

            else:
                dependencies.append(None)
                if channel_composition:
                    channel.dtype_fmt = get_fmt_v4(channel.data_type, channel.bit_count, channel.channel_type)
                    composition_dtype.append((channel.name, channel.dtype_fmt))

            # go to next channel of the current channel group
            ch_addr = channel.next_ch_addr

        return ch_cntr, neg_ch_cntr, composition, composition_dtype

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
                            return b""
                    address = data_list.next_dl_addr
                data = b"".join(data)
            elif blk_id == b"##CN":
                data = b""
            elif blk_id == b"##HL":
                hl = HeaderList(address=address, stream=stream)

                data = self._load_signal_data(
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
                data = self._load_signal_data(
                    address=group.signal_data[index], stream=self._file
                )
            elif group.data_location == v4c.LOCATION_MEMORY:
                data = group.signal_data[index]
            else:
                data = []
                stream = self._tempfile
                if group.signal_data[index]:
                    for addr, size in zip(
                        group.signal_data[index], group.signal_data_size[index]
                    ):
                        if not size:
                            continue
                        stream.seek(addr)
                        data.append(stream.read(size))
                data = b"".join(data)
        else:
            data = b""

        return data

    def _load_data(self, group, record_offset=0, record_count=None):
        """ get group's data block bytes """
        offset = 0
        has_yielded = False
        _count = record_count
        data_group = group.data_group
        channel_group = group.channel_group

        if group.data_location == v4c.LOCATION_ORIGINAL_FILE:
            stream = self._file
        else:
            stream = self._tempfile

        read = stream.read
        seek = stream.seek

        samples_size = (
            channel_group.samples_byte_nr + channel_group.invalidation_bytes_nr
        )

        record_offset *= samples_size

        finished = False
        if record_count is not None:
            record_count *= samples_size

        if not samples_size:
            yield b"", offset, _count
        else:

            if self._read_fragment_size:
                split_size = self._read_fragment_size // samples_size
                split_size *= samples_size
            else:
                channels_nr = len(group.channels)

                y_axis = CONVERT

                idx = searchsorted(CHANNEL_COUNT, channels_nr, side="right") - 1
                if idx < 0:
                    idx = 0
                split_size = y_axis[idx]

                split_size = split_size // samples_size
                split_size *= samples_size

            if split_size == 0:
                split_size = samples_size

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

                while True:
                    try:
                        info = next(blocks)
                        address, size, block_size, block_type, param = (
                            info.address,
                            info.raw_size,
                            info.size,
                            info.block_type,
                            info.param,
                        )
                        current_address = address
                    except StopIteration:
                        break

                    if group.sorted:
                        if offset + size < record_offset + 1:
                            offset += size
                            continue

                    seek(address)
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
                                if rec_id == record_id:
                                    rec_data.append(new_data[i : i + rec_size])
                            else:
                                (rec_size,) = UINT32_u(new_data[i : i + 4])
                                if rec_id == record_id:
                                    rec_data.append(new_data[i : i + 4 + rec_size])
                                i += 4
                            i += rec_size
                        new_data = b"".join(rec_data)

                        size = len(new_data)

                    if offset < record_offset:
                        delta = record_offset - offset
                        new_data = new_data[delta:]
                        size -= delta
                        offset = record_offset

                    while size >= split_size - cur_size:
                        if data:
                            data.append(new_data[:split_size - cur_size])
                            new_data = new_data[split_size - cur_size:]
                            data_ = b"".join(data)
                            if record_count is not None:
                                yield data_[:record_count], offset, _count
                                has_yielded = True
                                record_count -= len(data_)
                                if record_count <= 0:
                                    finished = True
                                    break
                            else:
                                yield data_, offset, _count
                                has_yielded = True
                            current_address += split_size - cur_size
                        else:
                            data_, new_data = new_data[:split_size], new_data[split_size:]
                            if record_count is not None:
                                yield data_[:record_count], offset, _count
                                has_yielded = True
                                record_count -= len(data_)
                                if record_count <= 0:
                                    finished = True
                                    break
                            else:
                                yield data_, offset, _count
                                has_yielded = True
                            current_address += split_size
                        offset += split_size

                        size -= split_size - cur_size
                        data = []
                        cur_size = 0

                    if finished:
                        data = []
                        break

                    if size:
                        data.append(new_data)
                        cur_size += size
                if data:
                    data_ = b"".join(data)
                    if record_count is not None:
                        yield data_[:record_count], offset, _count
                        has_yielded = True
                        record_count -= len(data_)
                    else:
                        yield data_, offset, _count
                        has_yielded = True

                if not has_yielded:
                    yield b"", 0, _count
            else:
                yield b"", offset, _count

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
        no_parent = None, None
        if parents is None:
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
                            if data_type not in v4c.NON_SCALAR_TYPES:
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

                                size = max(bit_count // 8, 1)
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
                    max_overlapping_size = (next_byte_aligned_position - start_offset) * 8
                    needed_size = bit_offset + bit_count
                    if max_overlapping_size >= needed_size:
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

            dtype_pair = "invalidation_bytes", "<u1", (invalidation_bytes_nr,)
            types.append(dtype_pair)

            dtypes = dtype(types)

            group.parents, group.types = parents, dtypes

        return parents, dtypes

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
        else:
            attachment_addr = 0

        # add channel block
        kwargs = {
            "channel_type": v4c.CHANNEL_TYPE_VALUE,
            "bit_count": signal.samples.dtype.itemsize * 8,
            "byte_offset": offset,
            "bit_offset": 0,
            "data_type": v4c.DATA_TYPE_BYTEARRAY,
            "precision": 255,
            "flags": 0,
        }
        if attachment_addr:
            kwargs["attachment_addr"] = attachment_addr
            kwargs["flags"] |= v4c.FLAG_CN_BUS_EVENT
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

            if sig_type == v4c.SIGNAL_TYPE_SCALAR:

                s_type, s_size = fmt_to_datatype_v4(samples.dtype, samples.shape)
                byte_size = s_size // 8

                fields.append(samples)
                types.append((field_name, samples.dtype, samples.shape[1:]))

                # add channel block
                kwargs = {
                    "channel_type": v4c.CHANNEL_TYPE_VALUE,
                    "bit_count": s_size,
                    "byte_offset": offset,
                    "bit_offset": 0,
                    "data_type": s_type,
                    "flags": 0,
                }

                if attachment_addr:
                    kwargs["flags"] |= v4c.FLAG_CN_BUS_EVENT

                if invalidation_bytes_nr:
                    if signal.invalidation_bits is not None:
                        inval_bits.append(signal.invalidation_bits)
                        kwargs["flags"] |= v4c.FLAG_CN_INVALIDATION_PRESENT
                        kwargs["pos_invalidation_bit"] = inval_cntr
                        inval_cntr += 1

                ch = Channel(**kwargs)
                ch.name = name

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

            elif sig_type == v4c.SIGNAL_TYPE_STRUCTURE_COMPOSITION:
                struct = Signal(
                    samples,
                    samples,
                    name=name,
                    invalidation_bits=signal.invalidation_bits,
                )
                offset, dg_cntr, ch_cntr, sub_structure, new_fields, new_types, inval_cntr = self._append_structure_composition(
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

    def _get_not_byte_aligned_data(self, data, group, ch_nr):
        big_endian_types = (
            v4c.DATA_TYPE_UNSIGNED_MOTOROLA,
            v4c.DATA_TYPE_REAL_MOTOROLA,
            v4c.DATA_TYPE_SIGNED_MOTOROLA,
        )

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
            bit_count = size << 3

        byte_count = bit_offset + bit_count
        if byte_count % 8:
            byte_count = (byte_count // 8) + 1
        else:
            byte_count //= 8

        types = [
            ("", f"a{byte_offset}"),
            ("vals", f"({byte_count},)u1"),
            ("", f"a{record_size - byte_count - byte_offset}"),
        ]

        vals = fromstring(data, dtype=dtype(types))

        vals = vals["vals"]

        if channel.data_type not in big_endian_types:
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

        if channel.data_type not in big_endian_types:
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
                ("vals", vals.dtype, vals.shape[1:]),
                ("", extra.dtype, extra.shape[1:]),
            ]
            vals = fromarrays([vals, extra], dtype=dtype(types))

        channel.dtype_fmt = get_fmt_v4(channel.data_type, bit_count, channel.channel_type)
        fmt = channel.dtype_fmt
        if size <= byte_count:
            if channel.data_type in big_endian_types:
                types = [("", f"a{byte_count - size}"), ("vals", fmt)]
            else:
                types = [("vals", fmt), ("", f"a{byte_count - size}")]
        else:
            types = [("vals", fmt)]

        vals = fromstring(vals.tobytes(), dtype=dtype(types))["vals"]

        if channel.data_type in v4c.SIGNED_INT:
            return as_non_byte_sized_signed_int(vals, bit_count)
        else:
            return vals

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
                        raise MdfException("Channel index out of range")
        else:
            if name not in self.channels_db:
                raise MdfException(f'Channel "{name}" not found')
            else:
                if source is not None:
                    for gp_nr, ch_nr in self.channels_db[name]:
                        source_name = self._get_source_name(gp_nr, ch_nr)
                        if source_name == source:
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
        return self.groups[group].channels[index].source.name or ""

    def _get_data_blocks_info(self, address, stream, block_type=b"##DT", mapped=False):
        info = []

        if mapped:
            if address:
                id_string, _, block_len, __ = COMMON_uf(stream, address)

                # can be a DataBlock
                if id_string == block_type:
                    size = block_len - 24
                    if size:
                        info.append(
                            DataBlockInfo(
                                address=address + COMMON_SIZE,
                                block_type=v4c.DT_BLOCK,
                                raw_size=size,
                                size=size,
                                param=0,
                            )
                        )
                # or a DataZippedBlock
                elif id_string == b"##DZ":
                    temp = {}
                    (
                        temp["id"],
                        temp["reserved0"],
                        temp["block_len"],
                        temp["links_nr"],
                        temp["original_type"],
                        temp["zip_type"],
                        temp["reserved1"],
                        temp["param"],
                        temp["original_size"],
                        temp["zip_size"],
                    ) = v4c.DZ_COMMON_uf(stream, address)

                    if temp["original_size"]:
                        if temp["zip_type"] == v4c.FLAG_DZ_DEFLATE:
                            block_type_ = v4c.DZ_BLOCK_DEFLATE
                            param = 0
                        else:
                            block_type_ = v4c.DZ_BLOCK_TRANSPOSED
                            param = temp["param"]
                        info.append(
                            DataBlockInfo(
                                address=address + v4c.DZ_COMMON_SIZE,
                                block_type=block_type_,
                                raw_size=temp["original_size"],
                                size=temp["zip_size"],
                                param=param,
                            )
                        )

                # or a DataList
                elif id_string == b"##DL":
                    while address:
                        dl = DataList(address=address, stream=stream, mapped=mapped)
                        for i in range(dl.data_block_nr):
                            addr = dl[f"data_block_addr{i}"]

                            id_string, _, block_len, __ = COMMON_uf(stream, addr)
                            # can be a DataBlock
                            if id_string == block_type:
                                size = block_len - 24
                                if size:
                                    info.append(
                                        DataBlockInfo(
                                            address=addr + COMMON_SIZE,
                                            block_type=v4c.DT_BLOCK,
                                            raw_size=size,
                                            size=size,
                                            param=0,
                                        )
                                    )
                            # or a DataZippedBlock
                            elif id_string == b"##DZ":
                                temp = {}
                                (
                                    temp["id"],
                                    temp["reserved0"],
                                    temp["block_len"],
                                    temp["links_nr"],
                                    temp["original_type"],
                                    temp["zip_type"],
                                    temp["reserved1"],
                                    temp["param"],
                                    temp["original_size"],
                                    temp["zip_size"],
                                ) = v4c.DZ_COMMON_uf(stream, addr)

                                if temp["original_size"]:
                                    if temp["zip_type"] == v4c.FLAG_DZ_DEFLATE:
                                        block_type_ = v4c.DZ_BLOCK_DEFLATE
                                        param = 0
                                    else:
                                        block_type_ = v4c.DZ_BLOCK_TRANSPOSED
                                        param = temp["param"]
                                    info.append(
                                        DataBlockInfo(
                                            address=addr + v4c.DZ_COMMON_SIZE,
                                            block_type=block_type_,
                                            raw_size=temp["original_size"],
                                            size=temp["zip_size"],
                                            param=param,
                                        )
                                    )
                        address = dl.next_dl_addr
                # or a header list
                elif id_string == b"##HL":
                    hl = HeaderList(address=address, stream=stream, mapped=mapped)
                    address = hl.first_dl_addr

                    info = self._get_data_blocks_info(
                        address,
                        stream,
                        block_type,
                        mapped,
                    )
        else:

            if address:
                stream.seek(address)
                id_string, _, block_len, __ = COMMON_u(stream.read(COMMON_SIZE))

                # can be a DataBlock
                if id_string == block_type:
                    size = block_len - 24
                    if size:
                        info.append(
                            DataBlockInfo(
                                address=address + COMMON_SIZE,
                                block_type=v4c.DT_BLOCK,
                                raw_size=size,
                                size=size,
                                param=0,
                            )
                        )
                # or a DataZippedBlock
                elif id_string == b"##DZ":
                    temp = {}
                    (
                        temp["id"],
                        temp["reserved0"],
                        temp["block_len"],
                        temp["links_nr"],
                        temp["original_type"],
                        temp["zip_type"],
                        temp["reserved1"],
                        temp["param"],
                        temp["original_size"],
                        temp["zip_size"],
                    ) = v4c.DZ_COMMON_u(stream, address)

                    if temp["original_size"]:
                        if temp["zip_type"] == v4c.FLAG_DZ_DEFLATE:
                            block_type_ = v4c.DZ_BLOCK_DEFLATE
                            param = 0
                        else:
                            block_type_ = v4c.DZ_BLOCK_TRANSPOSED
                            param = temp["param"]
                        info.append(
                            DataBlockInfo(
                                address=address + v4c.DZ_COMMON_SIZE,
                                block_type=block_type_,
                                raw_size=temp["original_size"],
                                size=temp["zip_size"],
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
                            id_string, _, block_len, __ = COMMON_u(
                                stream.read(COMMON_SIZE)
                            )
                            # can be a DataBlock
                            if id_string == block_type:
                                size = block_len - 24
                                if size:
                                    info.append(
                                        DataBlockInfo(
                                            address=addr + COMMON_SIZE,
                                            block_type=v4c.DT_BLOCK,
                                            raw_size=size,
                                            size=size,
                                            param=0,
                                        )
                                    )
                            # or a DataZippedBlock
                            elif id_string == b"##DZ":
                                temp = {}
                                (
                                    temp["id"],
                                    temp["reserved0"],
                                    temp["block_len"],
                                    temp["links_nr"],
                                    temp["original_type"],
                                    temp["zip_type"],
                                    temp["reserved1"],
                                    temp["param"],
                                    temp["original_size"],
                                    temp["zip_size"],
                                ) = v4c.DZ_COMMON_u(stream, address)

                                if temp["original_size"]:
                                    if temp["zip_type"] == v4c.FLAG_DZ_DEFLATE:
                                        block_type_ = v4c.DZ_BLOCK_DEFLATE
                                        param = 0
                                    else:
                                        block_type_ = v4c.DZ_BLOCK_TRANSPOSED
                                        param = temp["param"]
                                    info.append(
                                        DataBlockInfo(
                                            address=addr + v4c.DZ_COMMON_SIZE,
                                            block_type=block_type_,
                                            raw_size=temp["original_size"],
                                            size=temp["zip_size"],
                                            param=param,
                                        )
                                    )
                        address = dl.next_dl_addr
                # or a header list
                elif id_string == b"##HL":
                    hl = HeaderList(address=address, stream=stream)
                    address = hl.first_dl_addr

                    info = self._get_data_blocks_info(
                        address,
                        stream,
                        block_type,
                        mapped,
                    )

        return info

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
        invalidation_size = group.channel_group.invalidation_bytes_nr

        data_bytes, offset, _count = fragment
        try:
            invalidation = self._invalidation_cache[(group_index, offset, _count)]
        except KeyError:
            record = group.record
            if record is None:
                dtypes = group.types
                if dtypes.itemsize:
                    record = fromstring(data_bytes, dtype=dtypes)
                else:
                    record = None

            invalidation = record["invalidation_bytes"].tostring()
            self._invalidation_cache[(group_index, offset, _count)] = invalidation

        ch_invalidation_pos = channel["pos_invalidation_bit"]
        pos_byte, pos_offset = divmod(ch_invalidation_pos, 8)

        rec = fromstring(
            invalidation,
            dtype=[
                ("", f"S{pos_byte}"),
                ("vals", "<u1"),
                ("", f"S{invalidation_size - pos_byte - 1}"),
            ],
        )

        mask = 1 << pos_offset

        invalidation_bits = rec["vals"] & mask
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
            will contain the signal units mapped to the singal names when
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
        >>> s1 = Signal(samples=s1, timstamps=t, unit='+', name='Positive')
        >>> s2 = Signal(samples=s2, timstamps=t, unit='-', name='Negative')
        >>> s3 = Signal(samples=s3, timstamps=t, unit='flts', name='Floats')
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

        interp_mode = self._integer_interpolation

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
                    t = reduce(union1d, times).flatten().astype(float64)
                    signals = [s.interp(t, interpolation_mode=interp_mode) for s in signals]
                    times = None
                else:
                    t = t_
            else:
                t = t_
        else:
            t = []

        dg_cntr = len(self.groups)

        gp = Group(None)
        gp.signal_data = gp_sdata = []
        gp.signal_data_size = gp_sdata_size = []
        gp.channels = gp_channels = []
        gp.channel_dependencies = gp_dep = []
        gp.signal_types = gp_sig_types = []
        gp.logging_channels = []

        # channel group
        kwargs = {"cycles_nr": 0, "samples_byte_nr": 0}
        gp.channel_group = ChannelGroup(**kwargs)
        gp.channel_group.name = source_info

        if any(sig.invalidation_bits is not None for sig in signals):
            invalidation_bytes_nr = 1
            gp.channel_group.invalidation_bytes_nr = invalidation_bytes_nr

            inval_bits = []

        else:
            invalidation_bytes_nr = 0
            inval_bits = []
        inval_cntr = 0

        self.groups.append(gp)

        cycles_nr = len(t)
        fields = []
        types = []
        parents = {}
        ch_cntr = 0
        offset = 0
        field_names = UniqueDB()

        defined_texts = {}
        si_map = self._si_map
        cc_map = self._cc_map

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

        source_block = SourceInformation()
        source_block.name = source_block.path = source_info

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

                byte_size = max(s_size // 8, 1)

                if sig_dtype.kind == "u" and signal.bit_count <= 4:
                    s_size = signal.bit_count

                if signal.stream_sync:
                    channel_type = v4c.CHANNEL_TYPE_SYNC
                    at_data, at_name = signal.attachment
                    attachment_addr = self.attach(
                        at_data, at_name, mime="video/avi", embedded=False
                    )
                    data_block_addr = attachment_addr
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

                s_size = byte_size << 3

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
                offset, dg_cntr, ch_cntr, struct_self, new_fields, new_types, inval_cntr = self._append_structure_composition(
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

                if len(shape) > 1:
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
                gp_sdata_size.append(0)
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
                    byte_size = max(s_size // 8, 1)
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

                    gp_channels.append(ch)

                    entry = dg_cntr, ch_cntr
                    parent_dep.referenced_channels.append(entry)
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
                    gp_sdata.append([data_addr])
                    gp_sdata_size.append([data_size])
                    data.tofile(file)
                else:
                    data_addr = 0
                    gp_sdata.append([])
                    gp_sdata_size.append([])

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

            fields.append(inval_bits)
            types.append(("invalidation_bytes", inval_bits.dtype, inval_bits.shape[1:]))

        gp.channel_group.cycles_nr = cycles_nr
        gp.channel_group.samples_byte_nr = offset

        # data group
        gp.data_group = DataGroup()

        # data block
        types = dtype(types)

        gp.sorted = True
        gp.types = types
        gp.parents = parents

        if signals:
            samples = fromarrays(fields, dtype=types)
        else:
            samples = array([])

        signals = None
        del signals

        size = len(samples) * samples.itemsize

        if size:
            data_address = self._tempfile.tell()
            gp.data_location = v4c.LOCATION_TEMPORARY_FILE
            samples.tofile(self._tempfile)

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

        # channel group
        kwargs = {"cycles_nr": 0, "samples_byte_nr": 0}
        gp.channel_group = ChannelGroup(**kwargs)
        gp.channel_group.acq_name = source_info

        invalidation_bytes_nr = 0

        self.groups.append(gp)

        cycles_nr = len(t)
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

                byte_size = max(s_size // 8, 1)

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
                offsets = arange(len(sig), dtype=uint64) * (sig.itemsize + 4)

                values = [full(len(signal), sig.itemsize, dtype=uint32), sig]

                types_ = [("", uint32), ("", sig.dtype)]

                data = fromarrays(values, dtype=types_)

                data_size = len(data) * data.itemsize
                if data_size:
                    data_addr = tell()
                    gp_sdata.append([data_addr])
                    gp_sdata_size.append([data_size])
                    data.tofile(file)
                else:
                    data_addr = 0
                    gp_sdata.append([])
                    gp_sdata_size.append([])

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

                gp_channels.append(ch)

                offset += byte_size

                self.channels_db.add(name, (dg_cntr, ch_cntr))

                # update the parents as well
                field_name = field_names.field_names.get_unique_name(name)
                parents[ch_cntr] = field_name, 0

                fields.append(offsets)
                types.append((field_name, uint64))

                ch_cntr += 1

                # simple channels don't have channel dependencies
                gp_dep.append(None)

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
        >>> s1 = Signal(samples=s1, timstamps=t, unit='+', name='Positive')
        >>> s2 = Signal(samples=s2, timstamps=t, unit='-', name='Negative')
        >>> s3 = Signal(samples=s3, timstamps=t, unit='flts', name='Floats')
        >>> mdf = MDF3('new.mdf')
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

        fields = []
        types = []
        inval_bits = []

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
                cur_offset = sum(gp.signal_data_size[i])

                offsets = (
                    arange(len(signal), dtype=uint64) * (signal.itemsize + 4)
                    + cur_offset
                )
                values = [full(len(signal), signal.itemsize, dtype=uint32), signal]

                types_ = [("", uint32), ("", signal.dtype)]

                values = fromarrays(values, dtype=types_)

                stream.seek(0, 2)
                addr = stream.tell()
                block_size = len(values) * values.itemsize
                if block_size:
                    values.tofile(stream)
                    gp.signal_data[i].append(addr)
                    gp.signal_data_size[i].append(block_size)

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

            fields.append(inval_bits)
            types.append(("invalidation_bytes", inval_bits.dtype, inval_bits.shape[1:]))

        samples = fromarrays(fields, dtype=types)

        del fields
        del types

        stream.seek(0, 2)
        addr = stream.tell()
        size = len(samples) * samples.itemsize

        if size:
            samples.tofile(stream)

            gp.data_blocks.append(
                DataBlockInfo(
                    address=addr,
                    block_type=v4c.DT_BLOCK,
                    raw_size=size,
                    size=size,
                    param=0,
                )
            )

            record_size = gp.channel_group.samples_byte_nr
            record_size += gp.data_group.record_id_len
            record_size += gp.channel_group.invalidation_bytes_nr
            added_cycles = size // record_size
            gp.channel_group.cycles_nr += added_cycles

        del samples

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

                return data, file_path
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
                        return data, file_path
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
                    return data, file_path
        except Exception as err:
            os.chdir(current_path)
            message = "Exception during attachment extraction: " + repr(err)
            logger.warning(message)
            return b"", file_path

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
        copy_master=True,
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
        copy_master : bool
            make a copy of the timebase for this channel

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

        interp_mode = self._integer_interpolation

        if ch_nr >= 0:

            # get the channel object
            channel = grp.channels[ch_nr]

            dependency_list = grp.channel_dependencies[ch_nr]

            # get data group record
            parents, dtypes = grp.parents, grp.types
            if parents is None:
                parents, dtypes = self._prepare_record(grp)

            # get group data
            if data is None:
                data = self._load_data(
                    grp, record_offset=record_offset, record_count=record_count
                )
            else:
                data = (data,)

            channel_invalidation_present = (
                channel.flags
                & (v4c.FLAG_INVALIDATION_BIT_VALID | v4c.FLAG_ALL_SAMPLES_VALID)
                == v4c.FLAG_INVALIDATION_BIT_VALID
            )

            bit_count = channel.bit_count
        else:
            # get data group record
            parents, dtypes = self._prepare_record(grp)

            parent, bit_offset = parents[ch_nr]

            channel_invalidation_present = False
            dependency_list = None

            channel = grp.logging_channels[-ch_nr - 1]

            # get group data
            if data is None:
                data = self._load_data(
                    grp, record_offset=record_offset, record_count=record_count
                )
            else:
                data = (data,)

            bit_count = channel.bit_count

        data_type = channel.data_type
        channel_type = channel.channel_type
        stream_sync = channel_type == v4c.CHANNEL_TYPE_SYNC

        encoding = None

        master_is_required = not samples_only or raster

        # check if this is a channel array
        if dependency_list:
            if not isinstance(dependency_list[0], ChannelArrayBlock):
                # structure channel composition

                _dtype = dtype(channel.dtype_fmt)
                if _dtype.itemsize == bit_count // 8:
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
                            timestamps.append(
                                self.get_master(
                                    gp_nr, fragment, copy_master=copy_master
                                )
                            )
                        if channel_invalidation_present:
                            invalidation_bits.append(
                                self.get_invalidation_bits(gp_nr, channel, fragment)
                            )

                        count += 1
                else:
                    fast_path = False
                    names = [grp.channels[ch_nr].name for _, ch_nr in dependency_list]

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
                                raw=raw,
                                data=fragment,
                                ignore_invalidation_bits=ignore_invalidation_bits,
                                record_offset=record_offset,
                                record_count=record_count,
                            )[0]
                            channel_values[i].append(vals)
                        if master_is_required:
                            timestamps.append(
                                self.get_master(
                                    gp_nr, fragment, copy_master=copy_master
                                )
                            )
                        if channel_invalidation_present:
                            invalidation_bits.append(
                                self.get_invalidation_bits(gp_nr, channel, fragment)
                            )

                        count += 1

                if fast_path:
                    if count > 1:
                        vals = concatenate(channel_values)
                    else:
                        vals = channel_values[0]
                else:
                    if count > 1:
                        arrays = [concatenate(lst) for lst in channel_values]
                    else:
                        arrays = [lst[0] for lst in channel_values]
                    types = [
                        (name_, arr.dtype, arr.shape[1:])
                        for name_, arr in zip(names, arrays)
                    ]
                    types = dtype(types)

                    vals = fromarrays(arrays, dtype=types)

                if master_is_required:
                    if count > 1:
                        timestamps = concatenate(timestamps)
                    else:
                        timestamps = timestamps[0]

                if channel_invalidation_present:
                    if count > 1:
                        invalidation_bits = concatenate(invalidation_bits)
                    else:
                        invalidation_bits = invalidation_bits[0]
                    if not ignore_invalidation_bits:
                        vals = vals[nonzero(~invalidation_bits)[0]]
                        if master_is_required:
                            timestamps = timestamps[nonzero(~invalidation_bits)[0]]

                if raster and len(timestamps) > 1:
                    t = arange(timestamps[0], timestamps[-1], raster)

                    vals = (
                        Signal(vals, timestamps, name="_")
                        .interp(t, interpolation_mode=interp_mode)
                    )

                    vals, timestamps, invalidation_bits = (
                        vals.samples,
                        vals.timestamps,
                        vals.invalidation_bits,
                    )

            else:
                # channel arrays
                channel_group = grp.channel_group
                samples_size = (
                    channel_group.samples_byte_nr + channel_group.invalidation_bytes_nr
                )

                channel_values = []
                timestamps = []
                invalidation_bits = []
                count = 0
                for fragment in data:

                    data_bytes, offset, _count = fragment

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

                        vals = record[parent]
                    else:
                        vals = self._get_not_byte_aligned_data(data_bytes, grp, ch_nr)

                    vals = vals.copy()

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
                                    try:
                                        ref_dg_nr, ref_ch_nr = ca_block.referenced_channels[
                                            i
                                        ]
                                    except:
                                        debug_channel(
                                            self, grp, channel, dependency_list
                                        )
                                        raise

                                    axisname = (
                                        self.groups[ref_dg_nr].channels[ref_ch_nr].name
                                    )

                                    shape = (ca_block[f"dim_size_{i}"],)
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
                                        record_size += (
                                            channel_group.invalidation_bytes_nr
                                        )
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
                                ref_dg_nr, ref_ch_nr = ca_block.referenced_channels[i]

                                axisname = (
                                    self.groups[ref_dg_nr].channels[ref_ch_nr].name
                                )

                                shape = (ca_block[f"dim_size_{i}"],)
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

                                arrays.append(axis_values)
                                dtype_pair = axisname, axis_values.dtype, shape
                                types.append(dtype_pair)

                    vals = fromarrays(arrays, dtype(types))

                    if master_is_required:
                        timestamps.append(
                            self.get_master(gp_nr, fragment, copy_master=copy_master)
                        )
                    if channel_invalidation_present:
                        invalidation_bits.append(
                            self.get_invalidation_bits(gp_nr, channel, fragment)
                        )

                    channel_values.append(vals)
                    count += 1

                if count > 1:
                    vals = concatenate(channel_values)
                elif count == 1:
                    vals = channel_values[0]
                else:
                    vals = []

                if master_is_required:
                    if count > 1:
                        timestamps = concatenate(timestamps)
                    else:
                        timestamps = timestamps[0]

                if channel_invalidation_present:
                    if count > 1:
                        invalidation_bits = concatenate(invalidation_bits)
                    else:
                        invalidation_bits = invalidation_bits[0]
                    if not ignore_invalidation_bits:
                        vals = vals[nonzero(~invalidation_bits)[0]]
                        if master_is_required:
                            timestamps = timestamps[nonzero(~invalidation_bits)[0]]

                if raster and len(timestamps) > 1:
                    t = arange(timestamps[0], timestamps[-1], raster)

                    vals = (
                        Signal(vals, timestamps, name="_")
                        .interp(t, interpolation_mode=interp_mode)
                    )

                    vals, timestamps, invalidation_bits = (
                        vals.samples,
                        vals.timestamps,
                        vals.invalidation_bits,
                    )

            conversion = channel.conversion

        else:
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
                    data_bytes, offset, _count = fragment
                    offset = offset // record_size

                    vals = arange(len(data_bytes) // record_size, dtype=ch_dtype)
                    vals += offset

                    if master_is_required:
                        timestamps.append(
                            self.get_master(gp_nr, fragment, copy_master=copy_master)
                        )
                    if channel_invalidation_present:
                        invalidation_bits.append(
                            self.get_invalidation_bits(gp_nr, channel, fragment)
                        )

                    channel_values.append(vals)
                    count += 1

                if count > 1:
                    vals = concatenate(channel_values)
                elif count == 1:
                    vals = channel_values[0]
                else:
                    vals = []

                if master_is_required:
                    if count > 1:
                        timestamps = concatenate(timestamps)
                    else:
                        timestamps = timestamps[0]

                if channel_invalidation_present:
                    if count > 1:
                        invalidation_bits = concatenate(invalidation_bits)
                    else:
                        invalidation_bits = invalidation_bits[0]
                    if not ignore_invalidation_bits:
                        vals = vals[nonzero(~invalidation_bits)[0]]
                        if master_is_required:
                            timestamps = timestamps[nonzero(~invalidation_bits)[0]]

                if raster and len(timestamps) > 1:
                    num = float(float32((timestamps[-1] - timestamps[0]) / raster))
                    if num.is_integer():
                        t = linspace(timestamps[0], timestamps[-1], int(num))
                    else:
                        t = arange(timestamps[0], timestamps[-1], raster)

                    vals = (
                        Signal(vals, timestamps, name="_")
                        .interp(t, interpolation_mode=interp_mode)
                    )

                    vals, timestamps, invalidation_bits = (
                        vals.samples,
                        vals.timestamps,
                        vals.invalidation_bits,
                    )

            else:
                channel_values = []
                timestamps = []
                invalidation_bits = []

                count = 0
                for fragment in data:

                    data_bytes, offset, _count = fragment
                    try:
                        parent, bit_offset = parents[ch_nr]
                    except KeyError:
                        parent, bit_offset = None, None

                    if parent is not None:
                        if grp.record is None:
                            record = fromstring(data_bytes, dtype=dtypes)

                        else:
                            record = grp.record

                        vals = record[parent]

                        dtype_ = vals.dtype
                        shape_ = vals.shape
                        size = vals.dtype.itemsize
                        for dim in shape_[1:]:
                            size *= dim

                        kind_ = dtype_.kind

                        if kind_ == "b":
                            pass
                        elif len(shape_) > 1 and data_type != v4c.DATA_TYPE_BYTEARRAY:
                            vals = self._get_not_byte_aligned_data(
                                data_bytes, grp, ch_nr
                            )
                        elif kind_ not in "ui":
                            if bit_offset:
                                vals = self._get_not_byte_aligned_data(
                                    data_bytes, grp, ch_nr
                                )
                            else:
                                if bit_count != size * 8:
                                    if (
                                        bit_count % 8 == 0
                                        and size in (2, 4, 8)
                                        and data_type <= 3
                                    ):  # integer types
                                        vals = vals.view(f"<u{size}")
                                        if data_type in v4c.SIGNED_INT:
                                            vals = as_non_byte_sized_signed_int(
                                                vals, bit_count
                                            )
                                        else:
                                            mask = (1 << bit_count) - 1
                                            if vals.flags.writeable:
                                                vals &= mask
                                            else:
                                                vals = vals & mask
                                    else:
                                        vals = self._get_not_byte_aligned_data(
                                            data_bytes, grp, ch_nr
                                        )
                                else:
                                    if data_type <= 3:
                                        if not channel.dtype_fmt:
                                            channel.dtype_fmt = get_fmt_v4(data_type, bit_count, channel_type)
                                        channel_dtype = dtype(channel.dtype_fmt.split(')')[-1])
                                        vals = vals.view(channel_dtype)

                        else:
                            if data_type <= 3:
                                if dtype_.byteorder == ">":
                                    if bit_offset or bit_count != size << 3:
                                        vals = self._get_not_byte_aligned_data(
                                            data_bytes, grp, ch_nr
                                        )
                                else:
                                    if bit_offset:
                                        if kind_ == "i":
                                            vals = vals.astype(
                                                dtype(f"{dtype_.byteorder}u{size}")
                                            )
                                            vals >>= bit_offset
                                        else:
                                            vals = vals >> bit_offset

                                    if bit_count != size << 3:
                                        if data_type in v4c.SIGNED_INT:
                                            vals = as_non_byte_sized_signed_int(
                                                vals, bit_count
                                            )
                                        else:
                                            mask = (1 << bit_count) - 1
                                            if vals.flags.writeable:
                                                vals &= mask
                                            else:
                                                vals = vals & mask
                            else:
                                if bit_count != size * 8:
                                    vals = self._get_not_byte_aligned_data(
                                        data_bytes, grp, ch_nr
                                    )
                                else:
                                    if not channel.dtype_fmt:
                                        channel.dtype_fmt = get_fmt_v4(data_type, bit_count, channel_type)
                                    channel_dtype = dtype(channel.dtype_fmt.split(')')[-1])
                                    vals = vals.view(channel_dtype)

                    else:
                        vals = self._get_not_byte_aligned_data(data_bytes, grp, ch_nr)

                    if bit_count == 1 and self._single_bit_uint_as_bool:
                        vals = array(vals, dtype=bool)
                    else:
                        if not channel.dtype_fmt:
                            channel.dtype_fmt = get_fmt_v4(
                                data_type, bit_count, channel_type,
                            )
                        channel_dtype = dtype(channel.dtype_fmt.split(")")[-1])
                        if vals.dtype != channel_dtype:
                            vals = vals.astype(channel_dtype)

                    if master_is_required:
                        timestamps.append(
                            self.get_master(gp_nr, fragment, copy_master=copy_master)
                        )
                    if channel_invalidation_present:
                        invalidation_bits.append(
                            self.get_invalidation_bits(gp_nr, channel, fragment)
                        )
                    if vals.flags.writeable:
                        channel_values.append(vals)
                    else:
                        channel_values.append(vals.copy())
                    count += 1

                if count > 1:
                    vals = concatenate(channel_values)
                elif count == 1:
                    vals = channel_values[0]
                else:
                    vals = []

                if master_is_required:
                    if count > 1:
                        timestamps = concatenate(timestamps)
                    elif count == 1:
                        timestamps = timestamps[0]
                    else:
                        timestamps = []

                if channel_invalidation_present:
                    if count > 1:
                        invalidation_bits = concatenate(invalidation_bits)
                    elif count == 1:
                        invalidation_bits = invalidation_bits[0]
                    else:
                        invalidation_bits = []
                    if not ignore_invalidation_bits:
                        vals = vals[nonzero(~invalidation_bits)[0]]
                        if master_is_required:
                            timestamps = timestamps[nonzero(~invalidation_bits)[0]]

                if raster and len(timestamps) > 1:

                    num = float(float32((timestamps[-1] - timestamps[0]) / raster))
                    if num.is_integer():
                        t = linspace(timestamps[0], timestamps[-1], int(num))
                    else:
                        t = arange(timestamps[0], timestamps[-1], raster)

                    vals = (
                        Signal(vals, timestamps, name="_")
                        .interp(t, interpolation_mode=interp_mode)
                    )

                    vals, timestamps, invalidation_bits = (
                        vals.samples,
                        vals.timestamps,
                        vals.invalidation_bits,
                    )

            # get the channel conversion
            conversion = channel.conversion

            if channel_type == v4c.CHANNEL_TYPE_VLSD:
                signal_data = self._load_signal_data(group=grp, index=ch_nr)
                if signal_data:
                    values = []

                    vals = vals.tolist()

                    for offset in vals:
                        (str_size,) = UINT32_uf(signal_data, offset)
                        offset += 4
                        values.append(signal_data[offset : offset + str_size])

                    if data_type == v4c.DATA_TYPE_BYTEARRAY:

                        vals = array(values)
                        vals = vals.view(dtype=f"({vals.itemsize},)u1")

                    else:

                        vals = array(values)

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
                    # no VLSD signal data samples
                    vals = array([], dtype="S")
                    if data_type != v4c.DATA_TYPE_BYTEARRAY:

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

            elif channel_type in {
                v4c.CHANNEL_TYPE_VALUE,
                v4c.CHANNEL_TYPE_MLSD,
            } and (
                v4c.DATA_TYPE_STRING_LATIN_1
                <= data_type
                <= v4c.DATA_TYPE_STRING_UTF_16_BE
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
                        f'wrong data type "{data_type}" for string channel'
                    )

            # CANopen date
            if data_type == v4c.DATA_TYPE_CANOPEN_DATE:

                vals = vals.tostring()

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

                del arrays
                conversion = None

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

                del arrays

            if not raw:
                if conversion:
                    vals = conversion.convert(vals)
                    conversion = None

        if samples_only:
            if not channel_invalidation_present or not ignore_invalidation_bits:
                invalidation_bits = None
            res = vals, invalidation_bits
        else:
            # search for unit in conversion texts

            if name is None:
                name = channel.name

            unit = conversion and conversion.unit or channel.unit

            comment = channel.comment

            source = channel.source
            cg_source = grp.channel_group.acq_source
            if source:
                source = SignalSource(
                    source.name,
                    source.path,
                    source.comment,
                    source.source_type,
                    source.bus_type,
                )
            elif cg_source:
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
            elif channel_type == v4c.CHANNEL_TYPE_SYNC:
                index = self._attachments_map[channel.data_block_addr]
                attachment = self.extract_attachment(index=index)
            else:
                attachment = ()

            master_metadata = self._master_channel_metadata.get(gp_nr, None)

            if not channel_invalidation_present or not ignore_invalidation_bits:
                invalidation_bits = None

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
                    bit_count=bit_count,
                    stream_sync=stream_sync,
                    invalidation_bits=invalidation_bits,
                    encoding=encoding,
                )
            except:
                debug_channel(self, grp, channel, dependency_list)
                raise

        return res

    def get_master(
        self,
        index,
        data=None,
        raster=None,
        record_offset=0,
        record_count=None,
        copy_master=True,
    ):
        """ returns master channel samples for given group

        Parameters
        ----------
        index : int
            group index
        data : (bytes, int)
            (data block raw bytes, fragment offset); default None
        raster : float
            raster to be used for interpolation; default None
        record_offset : int
            if *data=None* use this to select the record offset from which the
            group data should be loaded
        record_count : int
            number of records to read; default *None* and in this case all
            available records are used
        copy_master : bool
            return a copy of the cached master

        Returns
        -------
        t : numpy.array
            master channel samples

        """
        fragment = data
        if fragment:
            data_bytes, offset, _count = fragment
            try:
                timestamps = self._master_channel_cache[(index, offset, _count)]
                if raster and len(timestamps):
                    timestamps = arange(timestamps[0], timestamps[-1], raster)
                    return timestamps
                else:
                    if copy_master:
                        return timestamps.copy()
                    else:
                        return timestamps
            except KeyError:
                pass
        else:
            try:
                timestamps = self._master_channel_cache[index]
                if raster and len(timestamps):
                    timestamps = arange(timestamps[0], timestamps[-1], raster)
                    return timestamps
                else:
                    if copy_master:
                        return timestamps.copy()
                    else:
                        return timestamps
            except KeyError:
                offset = 0

        group = self.groups[index]

        original_data = fragment

        if group.data_location == v4c.LOCATION_ORIGINAL_FILE:
            stream = self._file
        else:
            stream = self._tempfile

        time_ch_nr = self.masters_db.get(index, None)
        channel_group = group.channel_group
        record_size = channel_group.samples_byte_nr
        record_size += channel_group.invalidation_bytes_nr
        cycles_nr = group.channel_group.cycles_nr

        if original_data:
            cycles_nr = len(data_bytes) // record_size
        else:
            _count = record_count

        if time_ch_nr is None:
            if record_size:
                offset = offset // record_size
                t = arange(cycles_nr, dtype=float64)
                t += offset
            else:
                t = array([], dtype=float64)
            metadata = ("time", v4c.SYNC_TYPE_TIME)
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

            else:
                # get data group parents and dtypes
                parents, dtypes = group.parents, group.types
                if parents is None:
                    parents, dtypes = self._prepare_record(group)

                # get data
                if fragment is None:
                    data = self._load_data(
                        group, record_offset=record_offset, record_count=record_count
                    )
                else:
                    data = (fragment,)
                time_values = []

                for fragment in data:
                    data_bytes, offset, _count = fragment
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

                        t = record[parent]
                    else:
                        t = self._get_not_byte_aligned_data(
                            data_bytes, group, time_ch_nr
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
            data_bytes, offset, _ = original_data
            self._master_channel_cache[(index, offset, _count)] = t

        if raster and t.size:
            timestamps = t
            if len(t) > 1:
                num = float(float32((timestamps[-1] - timestamps[0]) / raster))
                if int(num) == num:
                    timestamps = linspace(t[0], t[-1], int(num))
                else:
                    timestamps = arange(t[0], t[-1], raster)
            return timestamps
        else:
            timestamps = t
            if copy_master:
                return timestamps.copy()
            else:
                return timestamps

    def get_can_signal(
        self, name, database=None, db=None, ignore_invalidation_bits=False
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

            if not database.lower().endswith(("dbc", "arxml")):
                message = f'Expected .dbc or .arxml file as CAN channel attachment but got "{database}"'
                logger.exception(message)
                raise MdfException(message)
            else:
                import_type = "dbc" if database.lower().endswith("dbc") else "arxml"
                with open(database, "rb") as db:
                    db_string = db.read()
                md5_sum = md5().update(db_string).digest()

                if md5_sum in self._external_dbc_cache:
                    db = self._external_dbc_cache[md5_sum]
                else:
                    try:
                        db_string = db_string.decode("utf-8")
                        db = self._external_dbc_cache[md5_sum] = loads(
                            db_string, importType=import_type, key="db"
                        )["db"]
                    except UnicodeDecodeError:
                        try:
                            from cchardet import detect

                            encoding = detect(db_string)["encoding"]
                            db_string = db_string.decode(encoding)
                            db = self._dbc_cache[md5_sum] = loads(
                                db_string,
                                importType=import_type,
                                key="db",
                                encoding=encoding,
                            )["db"]
                        except ImportError:
                            message = (
                                "Unicode exception occured while processing the database "
                                f'attachment "{database}" and "cChardet" package is '
                                'not installed. Mdf version 4 expects "utf-8" '
                                "strings and this package may detect if a different"
                                " encoding was used"
                            )
                            logger.warning(message)

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

        elif len(name_) == 2:
            message_id_str, signal = name_

            can_id = None

            message_id = v4c.CAN_DATA_FRAME_PATTERN.search(message_id_str)
            if message_id is None:
                message_id = message_id_str
            else:
                message_id = int(message_id)

        else:
            can_id = message_id = None
            signal = name

        if isinstance(message_id, str):
            message = db.frameByName(message_id)
        else:
            message = db.frameById(message_id)

        for sig in message.signals:
            if sig.name == signal:
                signal = sig
                break
        else:
            raise MdfException(
                f'Signal "{signal}" not found in message "{message.name}" of "{database}"'
            )

        if can_id is None:
            for _can_id, messages in self.can_logging_db.items():
                if message.id in messages:
                    index = messages[message.id]
                    break
            else:
                raise MdfException(
                    f'Message "{message.name}" (ID={hex(message.id)}) not found in the measurement'
                )
        else:
            if can_id in self.can_logging_db:
                if message.id in self.can_logging_db[can_id]:
                    index = self.can_logging_db[can_id][message.id]
                else:
                    raise MdfException(
                        f'Message "{message.name}" (ID={hex(message.id)}) not found in the measurement'
                    )
            else:
                raise MdfException(
                    f'No logging from "{can_id}" was found in the measurement'
                )

        can_ids = self.get(
            "CAN_DataFrame.ID",
            group=index,
            ignore_invalidation_bits=ignore_invalidation_bits,
        )
        payload = self.get(
            "CAN_DataFrame.DataBytes",
            group=index,
            samples_only=True,
            ignore_invalidation_bits=ignore_invalidation_bits,
        )[0]

        idx = nonzero(can_ids.samples == message.id)[0]
        vals = payload[idx]
        t = can_ids.timestamps[idx].copy()
        if can_ids.invalidation_bits is not None:
            invalidation_bits = can_ids.invalidation_bits
        else:
            invalidation_bits = None

        record_size = vals.shape[1]

        big_endian = False if signal.is_little_endian else True
        signed = signal.is_signed
        bit_offset = signal.startBit % 8
        byte_offset = signal.startBit // 8

        bit_count = signal.size

        byte_count = bit_offset + bit_count
        if byte_count % 8:
            byte_count = (byte_count // 8) + 1
        else:
            byte_count //= 8

        types = [
            ("", f"a{byte_offset}"),
            ("vals", f"({byte_count},)u1"),
            ("", f"a{record_size - byte_count - byte_offset}"),
        ]

        vals = vals.view(dtype=dtype(types))

        vals = vals["vals"]

        if not big_endian:
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

        if not big_endian:
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
                ("vals", vals.dtype, vals.shape[1:]),
                ("", extra.dtype, extra.shape[1:]),
            ]
            vals = fromarrays([vals, extra], dtype=dtype(types))

        fmt = "{}u{}".format(">" if big_endian else "<", size)
        if size <= byte_count:
            if big_endian:
                types = [("", f"a{byte_count - size}"), ("vals", fmt)]
            else:
                types = [("vals", fmt), ("", f"a{byte_count - size}")]
        else:
            types = [("vals", fmt)]

        vals = frombuffer(vals.tobytes(), dtype=dtype(types))

        if signed:
            vals = as_non_byte_sized_signed_int(vals["vals"], bit_count)
        else:
            vals = vals["vals"]

        comment = signal.comment or ""

        if (signal.factor, signal.offset) != (1, 0):
            vals = vals * float(signal.factor) + float(signal.offset)

        if ignore_invalidation_bits:

            return Signal(
                samples=vals,
                timestamps=t,
                name=name,
                unit=signal.unit or "",
                comment=comment,
                invalidation_bits=invalidation_bits,
            )

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
                        data_ = next(data)[0]
                        if compression and self.version > "4.00":
                            if compression == 1:
                                param = 0
                            else:
                                param = (
                                    gp.channel_group.samples_byte_nr
                                    + gp.channel_group.invalidation_bytes_nr
                                )
                            kwargs = {"data": data_, "zip_type": zip_type, "param": param}
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
                        kwargs = {"flags": v4c.FLAG_DL_EQUAL_LENGHT, "zip_type": zip_type}
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

                            if compression and self.version > "4.00":
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
                    if channel.attachment:
                        channel.attachment_addr = self.attachments[channel.attachment].address

                    address = channel.to_blocks(
                        address, blocks, defined_texts, cc_map, si_map
                    )

                    if channel.channel_type == v4c.CHANNEL_TYPE_SYNC:
                        idx = self._attachments_map[channel.data_block_addr]
                        channel.data_block_addr = self.attachments[idx].address
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
                                if dep.ca_type != v4c.CA_TYPE_LOOKUP:
                                    dep.referenced_channels = []
                                    continue
                                for i, (gp_nr, ch_nr) in enumerate(
                                    dep.referenced_channels
                                ):
                                    grp = self.groups[gp_nr]
                                    ch = grp.channels[ch_nr]
                                    dep[
                                        f"scale_axis_{i}_dg_addr"
                                    ] = grp.data_group.address
                                    dep[
                                        f"scale_axis_{i}_cg_addr"
                                    ] = grp.channel_group.address
                                    dep[f"scale_axis_{i}_ch_addr"] = ch.address

            for gp in self.groups:
                gp.data_group.record_id_len = 0

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
            self._master_channel_cache.clear()

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

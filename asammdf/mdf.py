# -*- coding: utf-8 -*-
""" common MDF file format module """

import csv
from datetime import datetime, timezone
from functools import reduce
import logging
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from copy import deepcopy
from struct import unpack
from shutil import copy
from pathlib import Path

import numpy as np
import pandas as pd

from .blocks.conversion_utils import from_dict
from .blocks.mdf_v2 import MDF2
from .blocks.mdf_v3 import MDF3
from .blocks.mdf_v4 import MDF4
from .signal import Signal
from .blocks.utils import (
    MdfException,
    matlab_compatible,
    validate_version_argument,
    MDF2_VERSIONS,
    MDF3_VERSIONS,
    MDF4_VERSIONS,
    SUPPORTED_VERSIONS,
    randomized_string,
    is_file_like,
    count_channel_groups,
    UINT16_u,
    UINT64_u,
    UniqueDB,
    components,
    downcast,
    master_using_raster,
    extract_can_signal,
    extract_mux,
    csv_int2hex,
    csv_bytearray2hex,
    load_can_database,
)

from .blocks.v2_v3_blocks import HeaderBlock as HeaderV3
from .blocks.v2_v3_blocks import ChannelConversion as ChannelConversionV3
from .blocks.v2_v3_blocks import ChannelExtension
from .blocks.v4_blocks import SourceInformation
from .blocks.v4_blocks import ChannelConversion as ChannelConversionV4
from .blocks.v4_blocks import HeaderBlock as HeaderV4
from .blocks.v4_blocks import EventBlock
from .blocks import v4_constants as v4c
from .blocks import v2_v3_constants as v23c


logger = logging.getLogger("asammdf")
LOCAL_TIMEZONE = datetime.now(timezone.utc).astimezone().tzinfo


__all__ = ["MDF", "SUPPORTED_VERSIONS"]


class MDF(object):
    """Unified access to MDF v3 and v4 files. Underlying _mdf's attributes and
    methods are linked to the `MDF` object via *setattr*. This is done to expose
    them to the user code and for performance considerations.

    Parameters
    ----------
    name : string | BytesIO
        mdf file name (if provided it must be a real file name) or
        file-like object

    version : string
        mdf file version from ('2.00', '2.10', '2.14', '3.00', '3.10', '3.20',
        '3.30', '4.00', '4.10', '4.11', '4.20'); default '4.10'


    callback (\*\*kwargs) : function
        keyword only argument: function to call to update the progress; the
        function must accept two arguments (the current progress and maximum
        progress value)
    use_display_names (\*\*kwargs) : bool
        keyword only argument: for MDF4 files parse the XML channel comment to
        search for the display name; XML parsing is quite expensive so setting
        this to *False* can decrease the loading times very much; default
        *False*
    remove_source_from_channel_names (\*\*kwargs) : bool
        remove source from channel names ("Speed\XCP3" -> "Speed")
    copy_on_get (\*\*kwargs) : bool
        copy arrays in the get method; default *True*

    """

    _terminate = False

    def __init__(self, name=None, version="4.10", **kwargs):
        if name:
            if is_file_like(name):
                file_stream = name
            else:
                name = Path(name)
                if name.is_file():
                    file_stream = open(name, "rb")
                else:
                    raise MdfException(f'File "{name}" does not exist')
            file_stream.seek(0)
            magic_header = file_stream.read(8)
            if magic_header != b"MDF     " and magic_header != b"UnFinMF ":
                raise MdfException(f'"{name}" is not a valid ASAM MDF file')
            file_stream.seek(8)
            version = file_stream.read(4).decode("ascii").strip(" \0")
            if not version:
                file_stream.read(16)
                version = unpack("<H", file_stream.read(2))[0]
                version = str(version)
                version = f"{version[0]}.{version[1:]}"
            if version in MDF3_VERSIONS:
                self._mdf = MDF3(name, **kwargs)
            elif version in MDF4_VERSIONS:
                self._mdf = MDF4(name, **kwargs)
            elif version in MDF2_VERSIONS:
                self._mdf = MDF2(name, **kwargs)
            else:
                message = f'"{name}" is not a supported MDF file; "{version}" file version was found'
                raise MdfException(message)

        else:
            version = validate_version_argument(version)
            if version in MDF2_VERSIONS:
                self._mdf = MDF3(version=version, **kwargs)
            elif version in MDF3_VERSIONS:
                self._mdf = MDF3(version=version, **kwargs)
            elif version in MDF4_VERSIONS:
                self._mdf = MDF4(version=version, **kwargs)
            else:
                message = (
                    f'"{version}" is not a supported MDF file version; '
                    f"Supported versions are {SUPPORTED_VERSIONS}"
                )
                raise MdfException(message)

        self._initial_attributes = set(dir(self))
        self._link_attributes()

    def _link_attributes(self):
        # link underlying _mdf attributes and methods to the new MDF object
        for attr in set(dir(self._mdf)) - self._initial_attributes:
            setattr(self, attr, getattr(self._mdf, attr))

        for attr in set(dir(self)) - set(dir(self._mdf)):
            if not attr.startswith("_"):
                setattr(self._mdf, attr, getattr(self, attr))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def __lt__(self, other):
        if self.header.start_time < other.header.start_time:
            return True
        elif self.header.start_time > other.header.start_time:
            return False
        else:

            t_min = []
            for i, group in enumerate(self.groups):
                cycles_nr = group.channel_group.cycles_nr
                if cycles_nr and i in self.masters_db:
                    master_min = self.get_master(i, record_offset=0, record_count=1)
                    if len(master_min):
                        t_min.append(master_min[0])

            other_t_min = []
            for i, group in enumerate(other.groups):
                cycles_nr = group.channel_group.cycles_nr
                if cycles_nr and i in other.masters_db:
                    master_min = other.get_master(i, record_offset=0, record_count=1)
                    if len(master_min):
                        other_t_min.append(master_min[0])

            if not t_min or not other_t_min:
                return True
            else:
                return min(t_min) < min(other_t_min)

    def _transfer_events(self, other):
        self._link_attributes()

        def get_scopes(event, events):
            if event.scopes:
                return event.scopes
            else:
                if event.parent is not None:
                    return get_scopes(events[event.parent], events)
                elif event.range_start is not None:
                    return get_scopes(events[event.range_start], events)
                else:
                    return event.scopes

        if other.version >= "4.00":
            for event in other.events:
                if self.version >= "4.00":
                    new_event = deepcopy(event)
                    event_valid = True
                    for i, ref in enumerate(new_event.scopes):
                        try:
                            dg_cntr, ch_cntr = ref
                            try:
                                (self.groups[dg_cntr].channels[ch_cntr])
                            except:
                                event_valid = False
                        except TypeError:
                            dg_cntr = ref
                            try:
                                (self.groups[dg_cntr].channel_group)
                            except:
                                event_valid = False
                    # ignore attachments for now
                    for i in range(new_event.attachment_nr):
                        key = f"attachment_{i}_addr"
                        event[key] = 0
                    if event_valid:
                        self.events.append(new_event)
                else:
                    ev_type = event.event_type
                    ev_range = event.range_type
                    ev_base = event.sync_base
                    ev_factor = event.sync_factor

                    timestamp = ev_base * ev_factor

                    try:
                        comment = ET.fromstring(
                            event.comment.replace(
                                ' xmlns="http://www.asam.net/mdf/v4"', ""
                            )
                        )
                        pre = comment.find(".//pre_trigger_interval")
                        if pre is not None:
                            pre = float(pre.text)
                        else:
                            pre = 0.0
                        post = comment.find(".//post_trigger_interval")
                        if post is not None:
                            post = float(post.text)
                        else:
                            post = 0.0
                        comment = comment.find(".//TX")
                        if comment is not None:
                            comment = comment.text
                        else:
                            comment = ""

                    except:
                        pre = 0.0
                        post = 0.0
                        comment = event.comment

                    if comment:
                        comment += ": "

                    if ev_range == v4c.EVENT_RANGE_TYPE_BEGINNING:
                        comment += "Begin of "
                    elif ev_range == v4c.EVENT_RANGE_TYPE_END:
                        comment += "End of "
                    else:
                        comment += "Single point "

                    if ev_type == v4c.EVENT_TYPE_RECORDING:
                        comment += "recording"
                    elif ev_type == v4c.EVENT_TYPE_RECORDING_INTERRUPT:
                        comment += "recording interrupt"
                    elif ev_type == v4c.EVENT_TYPE_ACQUISITION_INTERRUPT:
                        comment += "acquisition interrupt"
                    elif ev_type == v4c.EVENT_TYPE_START_RECORDING_TRIGGER:
                        comment += "measurement start trigger"
                    elif ev_type == v4c.EVENT_TYPE_STOP_RECORDING_TRIGGER:
                        comment += "measurement stop trigger"
                    elif ev_type == v4c.EVENT_TYPE_TRIGGER:
                        comment += "trigger"
                    else:
                        comment += "marker"

                    scopes = get_scopes(event, other.events)
                    if scopes:
                        for i, ref in enumerate(scopes):
                            event_valid = True
                            try:
                                dg_cntr, ch_cntr = ref
                                try:
                                    (self.groups[dg_cntr])
                                except:
                                    event_valid = False
                            except TypeError:
                                dg_cntr = ref
                                try:
                                    (self.groups[dg_cntr])
                                except:
                                    event_valid = False
                            if event_valid:

                                self.add_trigger(
                                    dg_cntr,
                                    timestamp,
                                    pre_time=pre,
                                    post_time=post,
                                    comment=comment,
                                )
                    else:
                        for i, _ in enumerate(self.groups):
                            self.add_trigger(
                                i,
                                timestamp,
                                pre_time=pre,
                                post_time=post,
                                comment=comment,
                            )

        else:
            for trigger_info in other.iter_get_triggers():
                comment = trigger_info["comment"]
                timestamp = trigger_info["time"]
                group = trigger_info["group"]

                if self.version < "4.00":
                    self.add_trigger(
                        group,
                        timestamp,
                        pre_time=trigger_info["pre_time"],
                        post_time=trigger_info["post_time"],
                        comment=comment,
                    )
                else:
                    if timestamp:
                        ev_type = v4c.EVENT_TYPE_TRIGGER
                    else:
                        ev_type = v4c.EVENT_TYPE_START_RECORDING_TRIGGER
                    event = EventBlock(
                        event_type=ev_type,
                        sync_base=int(timestamp * 10 ** 9),
                        sync_factor=10 ** -9,
                        scope_0_addr=0,
                    )
                    event.comment = comment
                    event.scopes.append(group)
                    self.events.append(event)

    def __contains__(self, channel):
        """ if *'channel name'* in *'mdf file'* """
        return channel in self.channels_db

    def __iter__(self):
        """ iterate over all the channels found in the file; master channels
        are skipped from iteration

        """
        yield from self.iter_channels()

    def convert(self, version):
        """convert *MDF* to other version

        Parameters
        ----------
        version : str
            new mdf file version from ('2.00', '2.10', '2.14', '3.00', '3.10',
            '3.20', '3.30', '4.00', '4.10', '4.11', '4.20'); default '4.10'

        Returns
        -------
        out : MDF
            new *MDF* object

        """
        self._link_attributes()
        version = validate_version_argument(version)

        out = MDF(version=version)

        out.header.start_time = self.header.start_time

        groups_nr = len(self.virtual_groups)

        if self._callback:
            self._callback(0, groups_nr)

        cg_nr = None

        self.configure(copy_on_get=False)

        # walk through all groups and get all channels
        for i, virtual_group in enumerate(self.virtual_groups):

            for idx, sigs in enumerate(self._yield_selected_signals(virtual_group, version=version)):
                if idx == 0:
                    source_info = f"Converted from {self.version} to {version}"
                    if sigs:
                        cg_nr = out.append(sigs, source_info, common_timebase=True)
                        out.groups[cg_nr].channel_group.comment = self.groups[virtual_group].channel_group.comment
                    else:
                        break
                else:
                    out.extend(cg_nr, sigs)

            if self._callback:
                self._callback(i + 1, groups_nr)

            if self._terminate:
                return

        out._transfer_events(self)
        self.configure(copy_on_get=True)
        if self._callback:
            out._callback = out._mdf._callback = self._callback
        return out

    def cut(
        self,
        start=None,
        stop=None,
        whence=0,
        version=None,
        include_ends=True,
        time_from_zero=False,
    ):
        """cut *MDF* file. *start* and *stop* limits are absolute values
        or values relative to the first timestamp depending on the *whence*
        argument.

        Parameters
        ----------
        start : float
            start time, default *None*. If *None* then the start of measurement
            is used
        stop : float
            stop time, default *None*. If *None* then the end of measurement is
            used
        whence : int
            how to search for the start and stop values

            * 0 : absolute
            * 1 : relative to first timestamp
        version : str
            new mdf file version from ('2.00', '2.10', '2.14', '3.00', '3.10',
            '3.20', '3.30', '4.00', '4.10', '4.11', 4.20'); default *None* and in this
            case the original file version is used
        include_ends : bool
            include the *start* and *stop* timestamps after cutting the signal.
            If *start* and *stop* are found in the original timestamps, then
            the new samples will be computed using interpolation. Default *True*
        time_from_zero : bool
            start time stamps from 0s in the cut measurement

        Returns
        -------
        out : MDF
            new MDF object

        """

        self._link_attributes()

        if version is None:
            version = self.version
        else:
            version = validate_version_argument(version)

        out = MDF(version=version)
        self.configure(copy_on_get=False)

        if whence == 1:
            timestamps = []
            for group in self.virtual_groups:
                master = self.get_master(group, record_offset=0, record_count=1)
                if master.size:
                    timestamps.append(master[0])

            if timestamps:
                first_timestamp = np.amin(timestamps)
            else:
                first_timestamp = 0

            if start is not None:
                start += first_timestamp
            if stop is not None:
                stop += first_timestamp

        if time_from_zero:
            delta = start
            t_epoch = self.header.start_time.timestamp() + delta
            out.header.start_time = datetime.fromtimestamp(t_epoch)
        else:
            delta = 0
            out.header.start_time = self.header.start_time

        groups_nr = len(self.virtual_groups)

        if self._callback:
            self._callback(0, groups_nr)

        interpolation_mode = self._integer_interpolation

        # walk through all groups and get all channels
        for i, (group_index, virtual_group) in enumerate(self.virtual_groups.items()):

            included_channels = self.included_channels(group_index)[group_index]
            if not included_channels:
                continue

            idx = 0
            for j, sigs in enumerate(self._yield_selected_signals(group_index, groups=included_channels)):

                if not sigs:
                    break
                if j == 0:
                    master = sigs[0].timestamps
                    signals = sigs
                else:
                    master = sigs[0][0]

                if not len(master):
                    continue

                needs_cutting = True

                # check if this fragement is within the cut interval or
                # if the cut interval has ended
                if start is None and stop is None:
                    fragment_start = None
                    fragment_stop = None
                    start_index = 0
                    stop_index = len(master)
                    needs_cutting = False
                elif start is None:
                    fragment_start = None
                    start_index = 0
                    if master[0] > stop:
                        break
                    else:
                        fragment_stop = min(stop, master[-1])
                        stop_index = np.searchsorted(
                            master, fragment_stop, side="right"
                        )
                        if stop_index == len(master):
                            needs_cutting = False

                elif stop is None:
                    fragment_stop = None
                    if master[-1] < start:
                        continue
                    else:
                        fragment_start = max(start, master[0])
                        start_index = np.searchsorted(
                            master, fragment_start, side="left"
                        )
                        stop_index = len(master)
                        if start_index == 0:
                            needs_cutting = False
                else:
                    if master[0] > stop:
                        break
                    elif master[-1] < start:
                        continue
                    else:
                        fragment_start = max(start, master[0])
                        start_index = np.searchsorted(
                            master, fragment_start, side="left"
                        )
                        fragment_stop = min(stop, master[-1])
                        stop_index = np.searchsorted(
                            master, fragment_stop, side="right"
                        )
                        if start_index == 0 and stop_index == len(master):
                            needs_cutting = False

                # update the signal is this is not the first yield
                if j:
                    for signal, (samples, invalidation) in zip (signals, sigs[1:]):
                        signal.samples = samples
                        signal.timestamps = master
                        signal.invalidation_bits = invalidation

                if needs_cutting:
                    master = (
                        Signal(master, master, name="_")
                        .cut(
                            fragment_start,
                            fragment_stop,
                            include_ends,
                            interpolation_mode=interpolation_mode,
                        )
                        .timestamps
                    )

                    if not len(master):
                        continue

                    signals = [
                        sig.cut(
                            master[0],
                            master[-1],
                            include_ends=include_ends,
                            interpolation_mode=interpolation_mode,
                        )
                        for sig in signals
                    ]
                else:
                    for sig in signals:
                        native = sig.samples.dtype.newbyteorder('=')
                        if sig.samples.dtype != native:
                            sig.samples = sig.samples.astype(native)

                if time_from_zero:
                    master = master - delta
                    for sig in signals:
                        sig.timestamps = master

                if idx == 0:

                    if start:
                        start_ = f"{start}s"
                    else:
                        start_ = "start of measurement"
                    if stop:
                        stop_ = f"{stop}s"
                    else:
                        stop_ = "end of measurement"
                    cg_nr = out.append(
                        signals,
                        f"Cut from {start_} to {stop_}",
                        common_timebase=True,
                    )
                    out.groups[cg_nr].channel_group.comment = self.groups[group_index].channel_group.comment

                else:
                    sigs = [
                        (sig.samples, sig.invalidation_bits)
                        for sig in signals
                    ]
                    sigs.insert(0, (master, None))
                    out.extend(cg_nr, sigs)

                idx += 1

            # if the cut interval is not found in the measurement
            # then append a data group with 0 cycles
            if idx == 0:
                for sig in signals:
                    sig.samples = sig.samples[:0]
                    sig.timestamps = sig.timestamps[:0]
                    if sig.invalidation_bits is not None:
                        sig.invaldiation_bits = sig.invalidation_bits[:0]

                if start:
                    start_ = f"{start}s"
                else:
                    start_ = "start of measurement"
                if stop:
                    stop_ = f"{stop}s"
                else:
                    stop_ = "end of measurement"
                out.append(
                    signals,
                    f"Cut from {start_} to {stop_}",
                    common_timebase=True,
                )

            if self._callback:
                self._callback(i + 1, groups_nr)

            if self._terminate:
                return

        self.configure(copy_on_get=True)

        out._transfer_events(self)
        if self._callback:
            out._callback = out._mdf._callback = self._callback
        return out

    def export(self, fmt, filename=None, **kwargs):
        """ export *MDF* to other formats. The *MDF* file name is used is
        available, else the *filename* argument must be provided.

        The *pandas* export option was removed. you should use the method
        *to_dataframe* instead.

        Parameters
        ----------
        fmt : string
            can be one of the following:

            * `csv` : CSV export that uses the "," delimiter. This option
              will generate a new csv file for each data group
              (<MDFNAME>_DataGroup_<cntr>.csv)

            * `hdf5` : HDF5 file output; each *MDF* data group is mapped to
              a *HDF5* group with the name 'DataGroup_<cntr>'
              (where <cntr> is the index)

            * `mat` : Matlab .mat version 4, 5 or 7.3 export. If
              *single_time_base==False* the channels will be renamed in the mat
              file to 'D<cntr>_<channel name>'. The channel group
              master will be renamed to 'DM<cntr>_<channel name>'
              ( *<cntr>* is the data group index starting from 0)

            * `parquet` : export to Apache parquet format

        filename : string | pathlib.Path
            export file name

        **kwargs

            * `single_time_base`: resample all channels to common time base,
              default *False*
            * `raster`: float time raster for resampling. Valid if
              *single_time_base* is *True*
            * `time_from_zero`: adjust time channel to start from 0
            * `use_display_names`: use display name instead of standard channel
              name, if available.
            * `empty_channels`: behaviour for channels without samples; the
              options are *skip* or *zeros*; default is *skip*
            * `format`: only valid for *mat* export; can be '4', '5' or '7.3',
              default is '5'
            * `oned_as`: only valid for *mat* export; can be 'row' or 'column'
            * `keep_arrays` : keep arrays and structure channels as well as the
              component channels. If *True* this can be very slow. If *False*
              only the component channels are saved, and their names will be
              prefixed with the parent channel.
            * reduce_memory_usage : bool
              reduce memory usage by converting all float columns to float32 and
              searching for minimum dtype that can reprezent the values found
              in integer columns; default *False*
            * compression : str
              compression to be used

              * for ``parquet`` : "GZIP" or "SANPPY"
              * for ``hfd5`` : "gzip", "lzf" or "szip"
              * for ``mat`` : bool

            * time_as_date (False) : bool
              export time as local timezone datetimee; only valid for CSV export

              .. versionadded:: 5.8.0

            * ignore_value2text_conversions (False) : bool
              valid only for the channels that have value to text conversions and
              if *raw=False*. If this is True then the raw numeric values will be
              used, and the conversion will not be applied.

              .. versionadded:: 5.8.0


        """

        self._link_attributes()

        header_items = (
            "date",
            "time",
            "author_field",
            "department_field",
            "project_field",
            "subject_field",
        )

        if fmt != "pandas" and filename is None and self.name is None:
            message = (
                "Must specify filename for export"
                "if MDF was created without a file name"
            )
            logger.warning(message)
            return

        single_time_base = kwargs.get("single_time_base", False)
        raster = kwargs.get("raster", 0)
        time_from_zero = kwargs.get("time_from_zero", True)
        use_display_names = kwargs.get("use_display_names", True)
        empty_channels = kwargs.get("empty_channels", "skip")
        format = kwargs.get("format", "5")
        oned_as = kwargs.get("oned_as", "row")
        reduce_memory_usage = kwargs.get("reduce_memory_usage", False)
        compression = kwargs.get("compression", "")
        time_as_date = kwargs.get("time_as_date", False)
        ignore_value2text_conversions = kwargs.get(
            "ignore_value2text_conversions", False
        )

        if compression == "SNAPPY":
            try:
                import snappy
            except ImportError:
                logger.warning(
                    "snappy compressor is not installed; compression will be set to GZIP"
                )
                compression = "GZIP"

        filename = Path(filename) if filename else self.name

        if fmt == "parquet":
            try:
                from fastparquet import write as write_parquet
            except ImportError:
                logger.warning(
                    "fastparquet not found; export to parquet is unavailable"
                )
                return

        elif fmt == "hdf5":
            try:
                from h5py import File as HDF5
            except ImportError:
                logger.warning("h5py not found; export to HDF5 is unavailable")
                return

        elif fmt == "mat":
            if format == "7.3":
                try:
                    from hdf5storage import savemat
                except ImportError:
                    logger.warning(
                        "hdf5storage not found; export to mat v7.3 is unavailable"
                    )
                    return
            else:
                try:
                    from scipy.io import savemat
                except ImportError:
                    logger.warning("scipy not found; export to mat is unavailable")
                    return

        elif fmt not in ("csv",):
            raise MdfException(f"Export to {fmt} is not implemented")

        name = ""

        if self._callback:
            self._callback(0, 100)

        if single_time_base or fmt == "parquet":
            df = self.to_dataframe(
                raster=raster,
                time_from_zero=time_from_zero,
                use_display_names=use_display_names,
                empty_channels=empty_channels,
                reduce_memory_usage=reduce_memory_usage,
                ignore_value2text_conversions=ignore_value2text_conversions,
            )
            units = OrderedDict()
            comments = OrderedDict()
            used_names = UniqueDB()

            dropped = {}

            groups_nr = len(self.groups)
            for i, grp in enumerate(self.groups):
                if self._terminate:
                    return

                for ch in grp.channels:

                    if use_display_names:
                        channel_name = ch.display_name or ch.name
                    else:
                        channel_name = ch.name

                    channel_name = used_names.get_unique_name(channel_name)

                    if hasattr(ch, "unit"):
                        unit = ch.unit
                        if ch.conversion:
                            unit = unit or ch.conversion.unit
                    else:
                        unit = ""
                    comment = ch.comment

                    units[channel_name] = unit
                    comments[channel_name] = comment

                if self._callback:
                    self._callback(i + 1, groups_nr * 2)

        if fmt == "hdf5":
            filename = filename.with_suffix(".hdf")

            if single_time_base:

                with HDF5(str(filename), "w") as hdf:
                    # header information
                    group = hdf.create_group(str(filename))

                    if self.version in MDF2_VERSIONS + MDF3_VERSIONS:
                        for item in header_items:
                            group.attrs[item] = self.header[item].replace(b"\0", b"")

                    # save each data group in a HDF5 group called
                    # "DataGroup_<cntr>" with the index starting from 1
                    # each HDF5 group will have a string attribute "master"
                    # that will hold the name of the master channel

                    count = len(df.columns)

                    for i, channel in enumerate(df):
                        samples = df[channel]
                        unit = units.get(channel, "")
                        comment = comments.get(channel, "")

                        if samples.dtype.kind == "O":
                            if isinstance(samples[0], np.ndarray):
                                samples = np.vstack(samples)
                            else:
                                continue

                        if compression:
                            dataset = group.create_dataset(
                                channel, data=samples, compression=compression,
                            )
                        else:
                            dataset = group.create_dataset(channel, data=samples)
                        unit = unit.replace("\0", "")
                        if unit:
                            dataset.attrs["unit"] = unit
                        comment = comment.replace("\0", "")
                        if comment:
                            dataset.attrs["comment"] = comment

                        if self._callback:
                            self._callback(i + 1 + count, count * 2)

            else:
                with HDF5(str(filename), "w") as hdf:
                    # header information
                    group = hdf.create_group(str(filename))

                    if self.version in MDF2_VERSIONS + MDF3_VERSIONS:
                        for item in header_items:
                            group.attrs[item] = self.header[item].replace(b"\0", b"")

                    # save each data group in a HDF5 group called
                    # "DataGroup_<cntr>" with the index starting from 1
                    # each HDF5 group will have a string attribute "master"
                    # that will hold the name of the master channel

                    groups_nr = len(self.virtual_groups)
                    for i, (group_index, virtual_group) in enumerate(self.virtual_groups.items()):
                        channels = self.included_channels(group_index)[group_index]

                        if not channels:
                            continue

                        names = UniqueDB()
                        if self._terminate:
                            return

                        if len(virtual_group.groups) == 1:
                            comment = self.groups[
                                virtual_group.groups[0]
                            ].channel_group.comment
                        else:
                            comment = 'Virtual group i'

                        group_name = r"/" + f"ChannelGroup_{i}"
                        group = hdf.create_group(group_name)

                        group.attrs["comment"] = comment

                        master_index = self.masters_db.get(group_index, -1)

                        if master_index >= 0:
                            group.attrs["master"] = self.groups[group_index].channels[master_index].name

                        channels = [
                            (None, gp_index, ch_index)
                            for gp_index, channel_indexes in channels.items()
                            for ch_index in channel_indexes
                        ]

                        channels = self.select(channels)

                        for j, sig in enumerate(channels):
                            if use_display_names:
                                name = sig.display_name or sig.name
                            else:
                                name = sig.name
                            name = name.replace("\\", "_").replace("/", "_")
                            name = names.get_unique_name(name)
                            if reduce_memory_usage:
                                sig.samples = downcast(sig.samples)
                            if compression:
                                dataset = group.create_dataset(
                                    name, data=sig.samples, compression=compression,
                                )
                            else:
                                dataset = group.create_dataset(
                                    name, data=sig.samples, dtype=sig.samples.dtype
                                )
                            unit = sig.unit.replace("\0", "")
                            if unit:
                                dataset.attrs["unit"] = unit
                            comment = sig.comment.replace("\0", "")
                            if comment:
                                dataset.attrs["comment"] = comment

                        if self._callback:
                            self._callback(i + 1, groups_nr)

        elif fmt == "csv":
            if single_time_base:
                filename = filename.with_suffix(".csv")
                message = f'Writing csv export to file "{filename}"'
                logger.info(message)

                if time_as_date:
                    index = (
                        pd.to_datetime(
                            df.index + self.header.start_time.timestamp(), unit="s"
                        )
                        .tz_localize("UTC")
                        .tz_convert(LOCAL_TIMEZONE)
                        .astype(str)
                    )
                    df.index = index
                    df.index.name = "timestamps"

                with open(filename, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)

                    if hasattr(self, "can_logging_db") and self.can_logging_db:

                        dropped = {}

                        for name_ in df.columns:
                            if name_.endswith("CAN_DataFrame.ID"):
                                dropped[name_] = pd.Series(
                                    csv_int2hex(df[name_].astype("<u4") & 0x1FFFFFFF),
                                    index=df.index,
                                )

                            elif name_.endswith("CAN_DataFrame.DataBytes"):
                                dropped[name_] = pd.Series(
                                    csv_bytearray2hex(df[name_]), index=df.index
                                )

                        df = df.drop(columns=list(dropped))
                        for name, s in dropped.items():
                            df[name] = s

                    names_row = [df.index.name, *df.columns]
                    writer.writerow(names_row)

                    vals = [df.index, *(df[name] for name in df)]

                    count = len(df.index)

                    if self._terminate:
                        return

                    for i, row in enumerate(zip(*vals)):
                        writer.writerow(row)

                        if self._callback:
                            self._callback(i + 1 + count, count * 2)

            else:

                filename = filename.with_suffix(".csv")

                gp_count = len(self.virtual_groups)
                for i, (group_index, virtual_group) in enumerate(self.virtual_groups.items()):

                    if self._terminate:
                        return

                    message = f"Exporting group {i+1} of {gp_count}"
                    logger.info(message)

                    if len(virtual_group.groups) == 1:
                        comment = self.groups[
                            virtual_group.groups[0]
                        ].channel_group.comment
                    else:
                        comment = ""

                    if comment:
                        for char in r" \/:":
                            comment = comment.replace(char, "_")
                        group_csv_name = (
                            filename.parent
                            / f"{filename.stem}.ChannelGroup_{i}_{comment}.csv"
                        )
                    else:
                        group_csv_name = (
                            filename.parent / f"{filename.stem}.ChannelGroup_{i}.csv"
                        )

                    df = self.get_group(
                        group_index,
                        raster=raster,
                        time_from_zero=time_from_zero,
                        use_display_names=use_display_names,
                        reduce_memory_usage=reduce_memory_usage,
                        ignore_value2text_conversions=ignore_value2text_conversions,
                    )

                    if time_as_date:
                        index = (
                            pd.to_datetime(
                                df.index + self.header.start_time.timestamp(), unit="s"
                            )
                            .tz_localize("UTC")
                            .tz_convert(LOCAL_TIMEZONE)
                            .astype(str)
                        )
                        df.index = index
                        df.index.name = "timestamps"

                    with open(group_csv_name, "w", newline="") as csvfile:
                        writer = csv.writer(csvfile)

                        if hasattr(self, "can_logging_db") and self.can_logging_db:

                            dropped = {}

                            for name_ in df.columns:
                                if name_.endswith("CAN_DataFrame.ID"):
                                    dropped[name_] = pd.Series(
                                        csv_int2hex(df[name_] & 0x1FFFFFFF),
                                        index=df.index,
                                    )

                                elif name_.endswith("CAN_DataFrame.DataBytes"):
                                    dropped[name_] = pd.Series(
                                        csv_bytearray2hex(df[name_]), index=df.index
                                    )

                            df = df.drop(columns=list(dropped))
                            for name_, s in dropped.items():
                                df[name_] = s

                        names_row = [df.index.name, *df.columns]
                        writer.writerow(names_row)

                        vals = [df.index, *(df[name] for name in df)]

                        count = len(df.index)

                        for i, row in enumerate(zip(*vals)):
                            writer.writerow(row)

                    if self._callback:
                        self._callback(i + 1, gp_count)

        elif fmt == "mat":

            filename = filename.with_suffix(".mat")

            if not single_time_base:
                mdict = {}

                master_name_template = "DGM{}_{}"
                channel_name_template = "DG{}_{}"
                used_names = UniqueDB()

                groups_nr = len(self.virtual_groups)

                for i, (group_index, virtual_group) in enumerate(self.virtual_groups.items()):
                    if self._terminate:
                        return

                    channels = self.included_channels(group_index)[group_index]

                    if not channels:
                        continue

                    channels = [
                        (None, gp_index, ch_index)
                        for gp_index, channel_indexes in channels.items()
                        for ch_index in channel_indexes
                    ]

                    channels = self.select(
                        channels,
                        ignore_value2text_conversions=ignore_value2text_conversions,
                    )

                    master = channels[0].copy()
                    master.samples = master.timestamps

                    channels.insert(0, master)

                    for j, sig in enumerate(channels):
                        if j == 0:
                            channel_name = master_name_template.format(i, "timestamps")
                        else:

                            if use_display_names:
                                channel_name = sig.display_name or sig.name
                            else:
                                channel_name = sig.name
                            channel_name = channel_name_template.format(i, channel_name)

                        channel_name = matlab_compatible(channel_name)
                        channel_name = used_names.get_unique_name(channel_name)

                        if sig.samples.dtype.names:
                            sig.samples.dtype.names = [
                                matlab_compatible(name)
                                for name in sig.samples.dtype.names
                            ]

                        mdict[channel_name] = sig.samples

                    if self._callback:
                        self._callback(i + 1, groups_nr + 1)

            else:
                used_names = UniqueDB()
                mdict = {}

                count = len(df.columns)

                for i, name in enumerate(df.columns):
                    channel_name = matlab_compatible(name)
                    channel_name = used_names.get_unique_name(channel_name)

                    mdict[channel_name] = df[name].values

                    if hasattr(mdict[channel_name].dtype, "categories"):
                        mdict[channel_name] = np.array(mdict[channel_name], dtype="S")

                    if self._callback:
                        self._callback(i + 1 + count, count * 2)

                mdict["timestamps"] = df.index.values

            if self._callback:
                self._callback(80, 100)
            if format == "7.3":

                savemat(
                    str(filename),
                    mdict,
                    long_field_names=True,
                    format="7.3",
                    delete_unused_variables=False,
                    oned_as=oned_as,
                    structured_numpy_ndarray_as_struct=True,
                )
            else:
                savemat(
                    str(filename),
                    mdict,
                    long_field_names=True,
                    oned_as=oned_as,
                    do_compression=bool(compression),
                )
            if self._callback:
                self._callback(100, 100)

        elif fmt == "parquet":
            filename = filename.with_suffix(".parquet")
            if compression:
                write_parquet(filename, df, compression=compression)
            else:
                write_parquet(filename, df)

        else:
            message = (
                'Unsopported export type "{}". '
                'Please select "csv", "excel", "hdf5", "mat" or "pandas"'
            )
            message.format(fmt)
            logger.warning(message)

    def filter(self, channels, version=None):
        """ return new *MDF* object that contains only the channels listed in
        *channels* argument

        Parameters
        ----------
        channels : list
            list of items to be filtered; each item can be :

                * a channel name string
                * (channel name, group index, channel index) list or tuple
                * (channel name, group index) list or tuple
                * (None, group index, channel index) list or tuple

        version : str
            new mdf file version from ('2.00', '2.10', '2.14', '3.00', '3.10',
            '3.20', '3.30', '4.00', '4.10', '4.11', '4.20'); default *None* and in this
            case the original file version is used

        Returns
        -------
        mdf : MDF
            new *MDF* file

        Examples
        --------
        >>> from asammdf import MDF, Signal
        >>> import numpy as np
        >>> t = np.arange(5)
        >>> s = np.ones(5)
        >>> mdf = MDF()
        >>> for i in range(4):
        ...     sigs = [Signal(s*(i*10+j), t, name='SIG') for j in range(1,4)]
        ...     mdf.append(sigs)
        ...
        >>> filtered = mdf.filter(['SIG', ('SIG', 3, 1), ['SIG', 2], (None, 1, 2)])
        >>> for gp_nr, ch_nr in filtered.channels_db['SIG']:
        ...     print(filtered.get(group=gp_nr, index=ch_nr))
        ...
        <Signal SIG:
                samples=[ 1.  1.  1.  1.  1.]
                timestamps=[0 1 2 3 4]
                unit=""
                info=None
                comment="">
        <Signal SIG:
                samples=[ 31.  31.  31.  31.  31.]
                timestamps=[0 1 2 3 4]
                unit=""
                info=None
                comment="">
        <Signal SIG:
                samples=[ 21.  21.  21.  21.  21.]
                timestamps=[0 1 2 3 4]
                unit=""
                info=None
                comment="">
        <Signal SIG:
                samples=[ 12.  12.  12.  12.  12.]
                timestamps=[0 1 2 3 4]
                unit=""
                info=None
                comment="">

        """

        self._link_attributes()

        if version is None:
            version = self.version
        else:
            version = validate_version_argument(version)

        # group channels by group index
        gps = self.included_channels(channels=channels)

        self.configure(copy_on_get=False)

        mdf = MDF(version=version)

        mdf.header.start_time = self.header.start_time

        if self.name:
            origin = self.name.parent
        else:
            origin = "New MDF"

        groups_nr = len(gps)

        if self._callback:
            self._callback(0, groups_nr)

        for i, (group_index, groups) in enumerate(gps.items()):

            for idx, sigs in enumerate(self._yield_selected_signals(group_index, groups=groups, version=version)):
                if not sigs:
                    break

                if idx == 0:

                    source_info = f"Signals filtered from <{origin}>"
                    if sigs:
                        cg_nr = mdf.append(sigs, source_info, common_timebase=True)
                        mdf.groups[cg_nr].channel_group.comment = self.groups[group_index].channel_group.comment
                    else:
                        break

                else:
                    mdf.extend(cg_nr, sigs)

            if self._callback:
                self._callback(i + 1, groups_nr)

            if self._terminate:
                return

        self.configure(copy_on_get=True)

        mdf._transfer_events(self)
        if self._callback:
            mdf._callback = mdf._mdf._callback = self._callback
        return mdf

    def iter_get(
        self,
        name=None,
        group=None,
        index=None,
        raster=None,
        samples_only=False,
        raw=False,
    ):
        """ iterator over a channel

        This is usefull in case of large files with a small number of channels.

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
        raw : bool
            return channel samples without appling the conversion rule; default
            `False`

        """
        self._link_attributes()

        gp_nr, ch_nr = self._validate_channel_selection(name, group, index)

        grp = self.groups[gp_nr]

        data = self._load_data(grp)

        for fragment in data:
            yield self.get(
                group=gp_nr,
                index=ch_nr,
                raster=raster,
                samples_only=samples_only,
                data=fragment,
                raw=raw,
            )

    @staticmethod
    def concatenate(
        files, version="4.10", sync=True, add_samples_origin=False, **kwargs
    ):
        """ concatenates several files. The files
        must have the same internal structure (same number of groups, and same
        channels in each group)

        Parameters
        ----------
        files : list | tuple
            list of *MDF* file names or *MDF* instances
        version : str
            merged file version
        sync : bool
            sync the files based on the start of measurement, default *True*
        add_samples_origin : bool
            option to create a new "__samples_origin" channel that will hold
            the index of the measurement from where each timestamp originated

        kwargs :

            use_display_names (False) : bool

        Returns
        -------
        concatenate : MDF
            new *MDF* object with concatenated channels

        Raises
        ------
        MdfException : if there are inconsistencies between the files

        """
        if not files:
            raise MdfException("No files given for merge")

        callback = kwargs.get("callback", None)
        if callback:
            callback(0, 100)

        mdf_nr = len(files)

        input_types = [isinstance(mdf, MDF) for mdf in files]
        use_display_names = kwargs.get("use_display_names", False)

        versions = []
        if sync:
            timestamps = []
            for file in files:
                if isinstance(file, MDF):
                    timestamps.append(file.header.start_time)
                    versions.append(file.version)
                else:
                    with open(file, "rb") as mdf:
                        mdf.seek(64)
                        blk_id = mdf.read(2)
                        if blk_id == b"HD":
                            header = HeaderV3
                            versions.append("3.00")
                        else:
                            versions.append("4.00")
                            blk_id += mdf.read(2)
                            if blk_id == b"##HD":
                                header = HeaderV4
                            else:
                                raise MdfException(f'"{file}" is not a valid MDF file')

                        header = header(address=64, stream=mdf)

                        timestamps.append(header.start_time)

            try:
                oldest = min(timestamps)
            except TypeError:
                timestamps = [
                    timestamp.astimezone(timezone.utc) for timestamp in timestamps
                ]
                oldest = min(timestamps)

            offsets = [(timestamp - oldest).total_seconds() for timestamp in timestamps]
            offsets = [offset if offset > 0 else 0 for offset in offsets]

        else:
            file = files[0]
            if isinstance(file, MDF):
                oldest = file.header.start_time
                versions.append(file.version)
            else:
                with open(file, "rb") as mdf:
                    mdf.seek(64)
                    blk_id = mdf.read(2)
                    if blk_id == b"HD":
                        versions.append("3.00")
                        header = HeaderV3
                    else:
                        versions.append("4.00")
                        blk_id += mdf.read(2)
                        if blk_id == b"##HD":
                            header = HeaderV4
                        else:
                            raise MdfException(f'"{file}" is not a valid MDF file')

                    header = header(address=64, stream=mdf)

                    oldest = header.start_time

            offsets = [0 for _ in files]

        version = validate_version_argument(version)

        merged = MDF(version=version, callback=callback)

        merged.header.start_time = oldest

        included_channel_names = []
        cg_map = {}

        if add_samples_origin:
            origin_conversion = {}
            for i, mdf in enumerate(files):
                origin_conversion[f"val_{i}"] = i
                if isinstance(mdf, MDF):
                    origin_conversion[f"text_{i}"] = str(mdf.name)
                else:
                    origin_conversion[f"text_{i}"] = str(mdf)
            origin_conversion = from_dict(origin_conversion)

        for mdf_index, (offset, mdf) in enumerate(zip(offsets, files)):
            if not isinstance(mdf, MDF):
                mdf = MDF(mdf, use_display_names=use_display_names)

            mdf.configure(copy_on_get=False)

            if mdf_index == 0:
                last_timestamps = [None for gp in mdf.virtual_groups]
                groups_nr = len(last_timestamps)

            else:
                if len(mdf.virtual_groups) != groups_nr:
                    raise MdfException(
                        f"internal structure of file <{mdf.name}> is different; different channel groups count"
                    )

            for i, group_index in enumerate(mdf.virtual_groups):
                included_channels = mdf.included_channels(group_index)[group_index]

                if mdf_index == 0:
                    included_channel_names.append(
                        [
                            mdf.groups[gp_index].channels[ch_index].name
                            for gp_index, channels in included_channels.items()
                            for ch_index in channels
                        ]
                    )
                else:
                    names = [
                        mdf.groups[gp_index].channels[ch_index].name
                        for gp_index, channels in included_channels.items()
                        for ch_index in channels
                    ]
                    if names != included_channel_names[i]:
                        if sorted(names) != sorted(included_channel_names[i]):
                            raise MdfException(
                                f"internal structure of file {mdf_index} is different; different channels order"
                            )
                        else:
                            raise MdfException(
                                f"internal structure of file {mdf_index} is different; different channels"
                            )
                if not included_channels:
                    continue

                idx = 0
                last_timestamp = last_timestamps[i]
                first_timestamp = None
                original_first_timestamp = None

                for idx, signals in enumerate(mdf._yield_selected_signals(group_index, groups=included_channels)):
                    if not signals:
                        break
                    if mdf_index == 0 and idx == 0:

                        first_signal = signals[0]
                        if len(first_signal):
                            if offset > 0:
                                timestamps = first_signal.timestamps + offset
                                for sig in signals:
                                    sig.timestamps = timestamps
                            last_timestamp = first_signal.timestamps[-1]
                            first_timestamp = first_signal.timestamps[0]
                            original_first_timestamp = first_timestamp

                        if add_samples_origin:
                            signals.append(
                                Signal(
                                    samples=np.ones(len(first_signal), dtype="<u2") * mdf_index,
                                    timestamps=first_signal.timestamps,
                                    conversion=origin_conversion,
                                    name="__samples_origin",
                                )
                            )

                        cg_nr = merged.append(signals, common_timebase=True)
                        merged.groups[cg_nr].channel_group.comment = mdf.groups[group_index].channel_group.comment
                        cg_map[group_index] = cg_nr

                    else:
                        if idx == 0:
                            signals = [(signals[0].timestamps, None)] + [
                                (sig.samples, sig.invalidation_bits)
                                for sig in signals
                            ]

                        master = signals[0][0]
                        _copied = False

                        if len(master):
                            if original_first_timestamp is None:
                                original_first_timestamp = master[0]
                            if offset > 0:
                                master = master + offset
                                _copied = True
                            if last_timestamp is None:
                                last_timestamp = master[-1]
                            else:
                                if last_timestamp >= master[0]:
                                    if len(master) >= 2:
                                        delta = master[1] - master[0]
                                    else:
                                        delta = 0.001
                                    if _copied:
                                        master -= master[0]
                                    else:
                                        master = master - master[0]
                                        _copied = True
                                    master += last_timestamp + delta
                                last_timestamp = master[-1]

                            signals[0] = master, None

                            if add_samples_origin:
                                signals.append(
                                    (
                                        np.ones(len(master), dtype="<u2") * mdf_index,
                                        None,
                                    )
                                )
                            cg_nr = cg_map[group_index]
                            merged.extend(cg_nr, signals)

                            if first_timestamp is None:
                                first_timestamp = master[0]

                last_timestamps[i] = last_timestamp

            mdf.configure(copy_on_get=True)

            if not input_types[mdf_index]:
                mdf.close()

            if callback:
                callback(i + 1 + mdf_index * groups_nr, groups_nr * mdf_nr)

            if MDF._terminate:
                return

            merged._transfer_events(mdf)

        return merged

    @staticmethod
    def stack(files, version="4.10", sync=True, **kwargs):
        """ stack several files and return the stacked *MDF* object

        Parameters
        ----------
        files : list | tuple
            list of *MDF* file names or *MDF* instances
        version : str
            merged file version
        sync : bool
            sync the files based on the start of measurement, default *True*

        kwargs :

            use_display_names (False) : bool

        Returns
        -------
        stacked : MDF
            new *MDF* object with stacked channels

        """
        if not files:
            raise MdfException("No files given for stack")

        version = validate_version_argument(version)

        callback = kwargs.get("callback", None)
        use_display_names = kwargs.get("use_display_names", False)

        stacked = MDF(version=version, callback=callback)

        files_nr = len(files)

        input_types = [isinstance(mdf, MDF) for mdf in files]

        if callback:
            callback(0, files_nr)

        if sync:
            timestamps = []
            for file in files:
                if isinstance(file, MDF):
                    timestamps.append(file.header.start_time)
                else:
                    with open(file, "rb") as mdf:
                        mdf.seek(64)
                        blk_id = mdf.read(2)
                        if blk_id == b"HD":
                            header = HeaderV3
                        else:
                            blk_id += mdf.read(2)
                            if blk_id == b"##HD":
                                header = HeaderV4
                            else:
                                raise MdfException(f'"{file}" is not a valid MDF file')

                        header = header(address=64, stream=mdf)

                        timestamps.append(header.start_time)

            try:
                oldest = min(timestamps)
            except TypeError:
                timestamps = [
                    timestamp.astimezone(timezone.utc) for timestamp in timestamps
                ]
                oldest = min(timestamps)

            offsets = [(timestamp - oldest).total_seconds() for timestamp in timestamps]

            stacked.header.start_time = oldest
        else:
            offsets = [0 for file in files]

        for mdf_index, (offset, mdf) in enumerate(zip(offsets, files)):
            if not isinstance(mdf, MDF):
                mdf = MDF(mdf, use_display_names=use_display_names)

            mdf.configure(copy_on_get=False)

            for i, group in enumerate(mdf.virtual_groups):
                dg_cntr = None
                included_channels = mdf.included_channels(group)[group]
                if not included_channels:
                    continue

                for idx, signals in enumerate(mdf._yield_selected_signals(group, groups=included_channels, version=version)):
                    if not signals:
                        break
                    if idx == 0:
                        if sync:
                            timestamps = signals[0].timestamps + offset
                            for sig in signals:
                                sig.timestamps = timestamps
                        dg_cntr = stacked.append(signals, common_timebase=True)
                        stacked.groups[dg_cntr].channel_group.comment = mdf.groups[group].channel_group.comment
                    else:
                        master = signals[0][0]
                        if sync:
                            master = master + offset
                            signals[0] = master, None

                        stacked.extend(dg_cntr, signals)

                if dg_cntr is not None:
                    for index in range(dg_cntr, len(stacked.groups)):

                        stacked.groups[
                            index
                        ].channel_group.comment = (
                            f'stacked from channel group {i} of "{mdf.name.parent}"'
                        )

            if callback:
                callback(mdf_index, files_nr)

            mdf.configure(copy_on_get=True)

            if not input_types[mdf_index]:
                mdf.close()

            if MDF._terminate:
                return

        return stacked

    def iter_channels(self, skip_master=True, copy_master=True):
        """ generator that yields a *Signal* for each non-master channel

        Parameters
        ----------
        skip_master : bool
            do not yield master channels; default *True*
        copy_master : bool
            copy master for each yielded channel

        """

        self._link_attributes()

        for index in self.virtual_groups:

            channels = [
                (None, gp_index, ch_index)
                for gp_index, channel_indexes in self.included_channels(index)[index].items()
                for ch_index in channel_indexes
            ]

            channels = self.select(channels, copy_master=copy_master)

            yield from channels

    def iter_groups(self):
        """ generator that yields channel groups as pandas DataFrames. If there
        are multiple occurences for the same channel name inside a channel
        group, then a counter will be used to make the names unique
        (<original_name>_<counter>)

        """

        self._link_attributes()

        for i in self.virtual_groups:
            yield self.get_group(i)

    def resample(self, raster, version=None, time_from_zero=False):
        """ resample all channels using the given raster. See *configure* to select
        the interpolation method for interger channels

        Parameters
        ----------
        raster : float | np.array | str
            new raster that can be

            * a float step value
            * a channel name who's timestamps will be used as raster (starting with asammdf 5.5.0)
            * an array (starting with asammdf 5.5.0)

        version : str
            new mdf file version from ('2.00', '2.10', '2.14', '3.00', '3.10',
            '3.20', '3.30', '4.00', '4.10', '4.11', '4.20'); default *None* and
            in this case the original file version is used

        time_from_zero : bool
            start time stamps from 0s in the cut measurement

        Returns
        -------
        mdf : MDF
            new *MDF* with resampled channels

        Examples
        --------
        >>> from asammdf import MDF, Signal
        >>> import numpy as np
        >>> mdf = MDF()
        >>> sig = Signal(name='S1', samples=[1,2,3,4], timestamps=[1,2,3,4])
        >>> mdf.append(sig)
        >>> sig = Signal(name='S2', samples=[1,2,3,4], timestamps=[1.1, 3.5, 3.7, 3.9])
        >>> mdf.append(sig)
        >>> resampled = mdf.resample(raster=0.1)
        >>> resampled.select(['S1', 'S2'])
        [<Signal S1:
                samples=[1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 4]
                timestamps=[1.  1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.  2.1 2.2 2.3 2.4 2.5 2.6 2.7
         2.8 2.9 3.  3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4. ]
                invalidation_bits=None
                unit=""
                conversion=None
                source=SignalSource(name='Python', path='Python', comment='', source_type=4, bus_type=0)
                comment=""
                mastermeta="('time', 1)"
                raw=True
                display_name=
                attachment=()>
        , <Signal S2:
                samples=[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 3 3 4 4]
                timestamps=[1.  1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.  2.1 2.2 2.3 2.4 2.5 2.6 2.7
         2.8 2.9 3.  3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4. ]
                invalidation_bits=None
                unit=""
                conversion=None
                source=SignalSource(name='Python', path='Python', comment='', source_type=4, bus_type=0)
                comment=""
                mastermeta="('time', 1)"
                raw=True
                display_name=
                attachment=()>
        ]
        >>> resampled = mdf.resample(raster='S2')
        >>> resampled.select(['S1', 'S2'])
        [<Signal S1:
                samples=[1 3 3 3]
                timestamps=[1.1 3.5 3.7 3.9]
                invalidation_bits=None
                unit=""
                conversion=None
                source=SignalSource(name='Python', path='Python', comment='', source_type=4, bus_type=0)
                comment=""
                mastermeta="('time', 1)"
                raw=True
                display_name=
                attachment=()>
        , <Signal S2:
                samples=[1 2 3 4]
                timestamps=[1.1 3.5 3.7 3.9]
                invalidation_bits=None
                unit=""
                conversion=None
                source=SignalSource(name='Python', path='Python', comment='', source_type=4, bus_type=0)
                comment=""
                mastermeta="('time', 1)"
                raw=True
                display_name=
                attachment=()>
        ]
        >>> resampled = mdf.resample(raster=[1.9, 2.0, 2.1])
        >>> resampled.select(['S1', 'S2'])
        [<Signal S1:
                samples=[1 2 2]
                timestamps=[1.9 2.  2.1]
                invalidation_bits=None
                unit=""
                conversion=None
                source=SignalSource(name='Python', path='Python', comment='', source_type=4, bus_type=0)
                comment=""
                mastermeta="('time', 1)"
                raw=True
                display_name=
                attachment=()>
        , <Signal S2:
                samples=[1 1 1]
                timestamps=[1.9 2.  2.1]
                invalidation_bits=None
                unit=""
                conversion=None
                source=SignalSource(name='Python', path='Python', comment='', source_type=4, bus_type=0)
                comment=""
                mastermeta="('time', 1)"
                raw=True
                display_name=
                attachment=()>
        ]
        >>> resampled = mdf.resample(raster='S2', time_from_zero=True)
        >>> resampled.select(['S1', 'S2'])
        [<Signal S1:
                samples=[1 3 3 3]
                timestamps=[0.  2.4 2.6 2.8]
                invalidation_bits=None
                unit=""
                conversion=None
                source=SignalSource(name='Python', path='Python', comment='', source_type=4, bus_type=0)
                comment=""
                mastermeta="('time', 1)"
                raw=True
                display_name=
                attachment=()>
        , <Signal S2:
                samples=[1 2 3 4]
                timestamps=[0.  2.4 2.6 2.8]
                invalidation_bits=None
                unit=""
                conversion=None
                source=SignalSource(name='Python', path='Python', comment='', source_type=4, bus_type=0)
                comment=""
                mastermeta="('time', 1)"
                raw=True
                display_name=
                attachment=()>
        ]
        """

        self._link_attributes()

        if version is None:
            version = self.version
        else:
            version = validate_version_argument(version)

        interpolation_mode = self._integer_interpolation

        mdf = MDF(version=version)

        mdf.header.start_time = self.header.start_time

        groups_nr = len(self.virtual_groups)

        if self._callback:
            self._callback(0, groups_nr)

        try:
            raster = float(raster)
            assert raster > 0
        except (TypeError, ValueError):
            if isinstance(raster, str):
                raster = self.get(raster).timestamps
            else:
                raster = np.array(raster)
        else:
            raster = master_using_raster(self, raster)

        if time_from_zero and len(raster):
            delta = raster[0]
            new_raster = raster - delta
            t_epoch = self.header.start_time.timestamp() + delta
            mdf.header.start_time = datetime.fromtimestamp(t_epoch)
        else:
            delta = 0
            new_raster = None
            mdf.header.start_time = self.header.start_time

        for i, (group_index, virtual_group) in enumerate(self.virtual_groups.items()):
            channels = [
                (None, gp_index, ch_index)
                for gp_index, channel_indexes in self.included_channels(group_index)[group_index].items()
                for ch_index in channel_indexes
            ]
            sigs = self.select(channels, raw=True)

            sigs = [
                sig.interp(raster, interpolation_mode=interpolation_mode)
                for sig in sigs
            ]

            if new_raster is not None :
                for sig in sigs:
                    if len(sig):
                        sig.timestamps = new_raster

            dg_cntr = mdf.append(sigs, common_timebase=True)
            mdf.groups[dg_cntr].channel_group.comment = self.groups[group_index].channel_group.comment

            if self._callback:
                self._callback(i + 1, groups_nr)

            if self._terminate:
                return

        if self._callback:
            self._callback(groups_nr, groups_nr)

        mdf._transfer_events(self)
        if self._callback:
            mdf._callback = mdf._mdf._callback = self._callback
        return mdf

    def select(
        self,
        channels,
        record_offset=0,
        raw=False,
        copy_master=True,
        ignore_value2text_conversions=False,
        record_count=None,
        validate=False,
    ):
        """ retrieve the channels listed in *channels* argument as *Signal*
        objects

        .. note:: the *dataframe* argument was removed in version 5.8.0
                  use the ``to_dataframe`` method instead

        Parameters
        ----------
        channels : list
            list of items to be filtered; each item can be :

                * a channel name string
                * (channel name, group index, channel index) list or tuple
                * (channel name, group index) list or tuple
                * (None, group index, channel index) list or tuple

        record_offset : int
            record number offset; optimization to get the last part of signal samples
        raw : bool
            get raw channel samples; default *False*
        copy_master : bool
            option to get a new timestamps array for each selected Signal or to
            use a shared array for channels of the same channel group; default *True*
        ignore_value2text_conversions (False) : bool
            valid only for the channels that have value to text conversions and
            if *raw=False*. If this is True then the raw numeric values will be
            used, and the conversion will not be applied.

            .. versionadded:: 5.8.0

        validate (False) : bool
            consider the invalidation bits

            .. versionadded:: 5.16.0

        Returns
        -------
        signals : list
            list of *Signal* objects based on the input channel list

        Examples
        --------
        >>> from asammdf import MDF, Signal
        >>> import numpy as np
        >>> t = np.arange(5)
        >>> s = np.ones(5)
        >>> mdf = MDF()
        >>> for i in range(4):
        ...     sigs = [Signal(s*(i*10+j), t, name='SIG') for j in range(1,4)]
        ...     mdf.append(sigs)
        ...
        >>> # select SIG group 0 default index 1 default, SIG group 3 index 1, SIG group 2 index 1 default and channel index 2 from group 1
        ...
        >>> mdf.select(['SIG', ('SIG', 3, 1), ['SIG', 2],  (None, 1, 2)])
        [<Signal SIG:
                samples=[ 1.  1.  1.  1.  1.]
                timestamps=[0 1 2 3 4]
                unit=""
                info=None
                comment="">
        , <Signal SIG:
                samples=[ 31.  31.  31.  31.  31.]
                timestamps=[0 1 2 3 4]
                unit=""
                info=None
                comment="">
        , <Signal SIG:
                samples=[ 21.  21.  21.  21.  21.]
                timestamps=[0 1 2 3 4]
                unit=""
                info=None
                comment="">
        , <Signal SIG:
                samples=[ 12.  12.  12.  12.  12.]
                timestamps=[0 1 2 3 4]
                unit=""
                info=None
                comment="">
        ]

        """

        self._link_attributes()

        virtual_groups = self.included_channels(channels=channels, minimal=False, skip_master=False)

        output_signals = {}

        for virtual_group, groups in virtual_groups.items():
            cycles_nr = self._mdf.virtual_groups[virtual_group].cycles_nr
            pairs = [
                (gp_index, ch_index)
                for gp_index, channel_indexes in groups.items()
                for ch_index in channel_indexes
            ]

            if record_count is None:
                cycles = cycles_nr - record_offset
            else:
                if cycles_nr < record_count + record_offset:
                    cycles = cycles_nr - record_offset
                else:
                    cycles = record_count

            signals = []

            current_pos = 0

            for idx, sigs in enumerate(self._yield_selected_signals(virtual_group, groups=groups, record_offset=record_offset, record_count=record_count)):
                if not sigs:
                    break
                if idx == 0:
                    next_pos = current_pos + len(sigs[0])

                    master = np.empty(cycles, dtype=sigs[0].timestamps.dtype)
                    master[current_pos:next_pos] = sigs[0].timestamps

                    for sig in sigs:
                        shape = (cycles,) + sig.samples.shape[1:]
                        signal = np.empty(shape, dtype=sig.samples.dtype)
                        signal[current_pos:next_pos] = sig.samples
                        sig.samples = signal
                        signals.append(sig)

                        if sig.invalidation_bits is not None:
                            inval = np.empty(cycles, dtype=sig.invalidation_bits.dtype)
                            inval[current_pos:next_pos] = sig.invalidation_bits
                            sig.invalidation_bits = inval

                else:
                    sig, _ = sigs[0]
                    next_pos = current_pos + len(sig)
                    master[current_pos:next_pos] = sig

                    for signal, (sig, inval) in zip(signals, sigs[1:]):
                        signal.samples[current_pos:next_pos] = sig
                        if signal.invalidation_bits is not None:
                            signal.invalidation_bits[current_pos:next_pos] = inval

                current_pos = next_pos

            for signal, pair in zip(signals, pairs):
                signal.timestamps = master
                output_signals[pair] = signal

        indexes = []

        for item in channels:
            if not isinstance(item, (list, tuple)):
                item = [item]
            indexes.append(self._validate_channel_selection(*item))

        signals = [output_signals[pair] for pair in indexes]

        if copy_master:
            for signal in signals:
                signal.timestamps = signal.timestamps.copy()

        if not raw:
            if ignore_value2text_conversions:
                for signal in signals:
                    conversion = signal.conversion
                    if conversion:
                        samples = conversion.convert(signal.samples)
                        if samples.dtype.kind not in 'US':
                            signal.samples = samples
                    signal.raw = True
                    signal.conversion = None
            else:
                for signal in signals:
                    conversion = signal.conversion
                    if conversion:
                        signal.samples = conversion.convert(signal.samples)
                    signal.raw = False
                    signal.conversion = None
                    if signal.samples.dtype.kind == 'S':
                        signal.encoding = 'utf-8' if self.version >= '4.00' else 'latin-1'

        if validate:
            signals = [sig.validate() for sig in signals]

        return signals

    def whereis(self, channel):
        """ get ocurrences of channel name in the file

        Parameters
        ----------
        channel : str
            channel name string

        Returns
        -------
        ocurrences : tuple


        Examples
        --------
        >>> mdf = MDF(file_name)
        >>> mdf.whereis('VehicleSpeed') # "VehicleSpeed" exists in the file
        ((1, 2), (2, 4))
        >>> mdf.whereis('VehicleSPD') # "VehicleSPD" doesn't exist in the file
        ()

        """
        self._link_attributes()

        if channel in self:
            return tuple(self.channels_db[channel])
        else:
            return tuple()

    @staticmethod
    def scramble(name, skip_attachments=False, **kwargs):
        """ scramble text blocks and keep original file structure

        Parameters
        ----------
        name : str | pathlib.Path
            file name
        skip_attachments : bool
            skip scrambling of attachments data if True

            .. versionadded:: 5.9.0

        Returns
        -------
        name : str
            scrambled file name

        """

        name = Path(name)

        mdf = MDF(name)
        texts = {}

        callback = kwargs.get("callback", None)
        if callback:
            callback(0, 100)

        count = len(mdf.groups)

        if mdf.version >= "4.00":
            ChannelConversion = ChannelConversionV4

            stream = mdf._file

            if mdf.header.comment_addr:
                stream.seek(mdf.header.comment_addr + 8)
                size = UINT64_u(stream.read(8))[0] - 24
                texts[mdf.header.comment_addr] = randomized_string(size)

            for fh in mdf.file_history:
                addr = fh.comment_addr
                if addr and addr not in texts:
                    stream.seek(addr + 8)
                    size = UINT64_u(stream.read(8))[0] - 24
                    texts[addr] = randomized_string(size)

            for ev in mdf.events:
                for addr in (ev.comment_addr, ev.name_addr):
                    if addr and addr not in texts:
                        stream.seek(addr + 8)
                        size = UINT64_u(stream.read(8))[0] - 24
                        texts[addr] = randomized_string(size)

            for at in mdf.attachments:
                for addr in (at.comment_addr, at.file_name_addr):
                    if addr and addr not in texts:
                        stream.seek(addr + 8)
                        size = UINT64_u(stream.read(8))[0] - 24
                        texts[addr] = randomized_string(size)
                if not skip_attachments and at.embedded_data:
                    texts[at.address + v4c.AT_COMMON_SIZE] = randomized_string(
                        at.embedded_size
                    )

            for idx, gp in enumerate(mdf.groups, 1):

                addr = gp.data_group.comment_addr
                if addr and addr not in texts:
                    stream.seek(addr + 8)
                    size = UINT64_u(stream.read(8))[0] - 24
                    texts[addr] = randomized_string(size)

                cg = gp.channel_group
                for addr in (cg.acq_name_addr, cg.comment_addr):
                    if cg.flags & v4c.FLAG_CG_BUS_EVENT:
                        continue

                    if addr and addr not in texts:
                        stream.seek(addr + 8)
                        size = UINT64_u(stream.read(8))[0] - 24
                        texts[addr] = randomized_string(size)

                    source = cg.acq_source_addr
                    if source:
                        source = SourceInformation(address=source, stream=stream)
                        for addr in (
                            source.name_addr,
                            source.path_addr,
                            source.comment_addr,
                        ):
                            if addr and addr not in texts:
                                stream.seek(addr + 8)
                                size = UINT64_u(stream.read(8))[0] - 24
                                texts[addr] = randomized_string(size)

                for ch in gp.channels:

                    for addr in (ch.name_addr, ch.unit_addr, ch.comment_addr):
                        if addr and addr not in texts:
                            stream.seek(addr + 8)
                            size = UINT64_u(stream.read(8))[0] - 24
                            texts[addr] = randomized_string(size)

                    source = ch.source_addr
                    if source:
                        source = SourceInformation(address=source, stream=stream)
                        for addr in (
                            source.name_addr,
                            source.path_addr,
                            source.comment_addr,
                        ):
                            if addr and addr not in texts:
                                stream.seek(addr + 8)
                                size = UINT64_u(stream.read(8))[0] - 24
                                texts[addr] = randomized_string(size)

                    conv = ch.conversion_addr
                    if conv:
                        conv = ChannelConversion(address=conv, stream=stream)
                        for addr in (conv.name_addr, conv.unit_addr, conv.comment_addr):
                            if addr and addr not in texts:
                                stream.seek(addr + 8)
                                size = UINT64_u(stream.read(8))[0] - 24
                                texts[addr] = randomized_string(size)
                        if conv.conversion_type == v4c.CONVERSION_TYPE_ALG:
                            addr = conv.formula_addr
                            if addr and addr not in texts:
                                stream.seek(addr + 8)
                                size = UINT64_u(stream.read(8))[0] - 24
                                texts[addr] = randomized_string(size)

                        if conv.referenced_blocks:
                            for key, block in conv.referenced_blocks.items():
                                if block:
                                    if isinstance(block, bytes):
                                        addr = conv[key]
                                        if addr not in texts:
                                            stream.seek(addr + 8)
                                            size = len(block)
                                            texts[addr] = randomized_string(size)

                if callback:
                    callback(int(idx / count * 66), 100)

            mdf.close()

            dst = name.with_suffix(".scrambled.mf4")

            copy(name, dst)

            with open(dst, "rb+") as mdf:
                count = len(texts)
                chunk = max(count // 34, 1)
                idx = 0
                for index, (addr, bts) in enumerate(texts.items()):
                    mdf.seek(addr + 24)
                    mdf.write(bts)
                    if index % chunk == 0:
                        if callback:
                            callback(66 + idx, 100)

            if callback:
                callback(100, 100)

        else:
            ChannelConversion = ChannelConversionV3

            stream = mdf._file

            if mdf.header.comment_addr:
                stream.seek(mdf.header.comment_addr + 2)
                size = UINT16_u(stream.read(2))[0] - 4
                texts[mdf.header.comment_addr + 4] = randomized_string(size)
            texts[36 + 0x40] = randomized_string(32)
            texts[68 + 0x40] = randomized_string(32)
            texts[100 + 0x40] = randomized_string(32)
            texts[132 + 0x40] = randomized_string(32)

            for idx, gp in enumerate(mdf.groups, 1):

                cg = gp.channel_group
                addr = cg.comment_addr

                if addr and addr not in texts:
                    stream.seek(addr + 2)
                    size = UINT16_u(stream.read(2))[0] - 4
                    texts[addr + 4] = randomized_string(size)

                if gp.trigger:
                    addr = gp.trigger.text_addr
                    if addr:
                        stream.seek(addr + 2)
                        size = UINT16_u(stream.read(2))[0] - 4
                        texts[addr + 4] = randomized_string(size)

                for ch in gp.channels:

                    for key in ("long_name_addr", "display_name_addr", "comment_addr"):
                        if hasattr(ch, key):
                            addr = getattr(ch, key)
                        else:
                            addr = 0
                        if addr and addr not in texts:
                            stream.seek(addr + 2)
                            size = UINT16_u(stream.read(2))[0] - 4
                            texts[addr + 4] = randomized_string(size)

                    texts[ch.address + 26] = randomized_string(32)
                    texts[ch.address + 58] = randomized_string(128)

                    source = ch.source_addr
                    if source:
                        source = ChannelExtension(address=source, stream=stream)
                        if source.type == v23c.SOURCE_ECU:
                            texts[source.address + 12] = randomized_string(80)
                            texts[source.address + 92] = randomized_string(32)
                        else:
                            texts[source.address + 14] = randomized_string(36)
                            texts[source.address + 50] = randomized_string(36)

                    conv = ch.conversion_addr
                    if conv:
                        texts[conv + 22] = randomized_string(20)

                        conv = ChannelConversion(address=conv, stream=stream)

                        if conv.conversion_type == v23c.CONVERSION_TYPE_FORMULA:
                            texts[conv + 36] = randomized_string(conv.block_len - 36)

                        if conv.referenced_blocks:
                            for key, block in conv.referenced_blocks.items():
                                if block:
                                    if isinstance(block, bytes):
                                        addr = conv[key]
                                        if addr and addr not in texts:
                                            stream.seek(addr + 2)
                                            size = UINT16_u(stream.read(2))[0] - 4
                                            texts[addr + 4] = randomized_string(size)
                if callback:
                    callback(int(idx / count * 66), 100)

            mdf.close()

            dst = name.with_suffix(".scrambled.mdf")

            copy(name, dst)

            with open(dst, "rb+") as mdf:
                chunk = count // 34
                idx = 0
                for index, (addr, bts) in enumerate(texts.items()):
                    mdf.seek(addr)
                    mdf.write(bts)
                    if chunk and index % chunk == 0:
                        if callback:
                            callback(66 + idx, 100)

            if callback:
                callback(100, 100)

        return dst

    def get_group(
        self,
        index,
        channels=None,
        raster=None,
        time_from_zero=True,
        empty_channels="skip",
        keep_arrays=False,
        use_display_names=False,
        time_as_date=False,
        reduce_memory_usage=False,
        raw=False,
        ignore_value2text_conversions=False,
        only_basenames=False,
    ):
        """ get channel group as pandas DataFrames. If there are multiple
        occurences for the same channel name, then a counter will be used to
        make the names unique (<original_name>_<counter>)

        Parameters
        ----------
        index : int
            channel group index
        use_display_names : bool
            use display name instead of standard channel name, if available.
        reduce_memory_usage : bool
            reduce memory usage by converting all float columns to float32 and
            searching for minimum dtype that can reprezent the values found
            in integer columns; default *False*
        raw (False) : bool
            the dataframe will contain the raw channel values

            .. versionadded:: 5.7.0

        ignore_value2text_conversions (False) : bool
            valid only for the channels that have value to text conversions and
            if *raw=False*. If this is True then the raw numeric values will be
            used, and the conversion will not be applied.

            .. versionadded:: 5.8.0

        keep_arrays (False) : bool
            keep arrays and structure channels as well as the
            component channels. If *True* this can be very slow. If *False*
            only the component channels are saved, and their names will be
            prefixed with the parent channel.

            .. versionadded:: 5.8.0

        empty_channels ("skip") : str
            behaviour for channels without samples; the options are *skip* or
            *zeros*; default is *skip*

            .. versionadded:: 5.8.0

        only_basenames (False) : bool
            use just the field names, without prefix, for structures and channel
            arrays

            .. versionadded:: 5.13.0

        Returns
        -------
        df : pandas.DataFrame

        """

        self._link_attributes()

        channels = [
            (None, gp_index, ch_index)
            for gp_index, channel_indexes in self.included_channels(index)[index].items()
            for ch_index in channel_indexes
        ]

        return self.to_dataframe(
            channels=channels,
            raster=raster,
            time_from_zero=time_from_zero,
            empty_channels="skip",
            keep_arrays=False,
            use_display_names=use_display_names,
            time_as_date=time_as_date,
            reduce_memory_usage=reduce_memory_usage,
            raw=raw,
            ignore_value2text_conversions=ignore_value2text_conversions,
            only_basenames=only_basenames,
        )

    def iter_to_dataframe(
        self,
        channels=None,
        raster=None,
        time_from_zero=True,
        empty_channels="skip",
        keep_arrays=False,
        use_display_names=False,
        time_as_date=False,
        reduce_memory_usage=False,
        raw=False,
        ignore_value2text_conversions=False,
        use_interpolation=True,
        only_basenames=False,
        chunk_ram_size=200 * 1024 * 1024,
        interpolate_outwards_with_nan=False,
    ):
        """ generator that yields pandas DataFrame's that should not exceed
        200MB of RAM

        .. versionadded:: 5.15.0

        Parameters
        ----------
        channels : list
            filter a subset of channels; default *None*
        raster : float | np.array | str
            new raster that can be

            * a float step value
            * a channel name who's timestamps will be used as raster (starting with asammdf 5.5.0)
            * an array (starting with asammdf 5.5.0)

            see `resample` for examples of urisng this argument

        time_from_zero : bool
            adjust time channel to start from 0; default *True*
        empty_channels : str
            behaviour for channels without samples; the options are *skip* or
            *zeros*; default is *skip*
        use_display_names : bool
            use display name instead of standard channel name, if available.
        keep_arrays : bool
            keep arrays and structure channels as well as the
            component channels. If *True* this can be very slow. If *False*
            only the component channels are saved, and their names will be
            prefixed with the parent channel.
        time_as_date : bool
            the dataframe index will contain the datetime timestamps
            according to the measurement start time; default *False*. If
            *True* then the argument ``time_from_zero`` will be ignored.
        reduce_memory_usage : bool
            reduce memory usage by converting all float columns to float32 and
            searching for minimum dtype that can reprezent the values found
            in integer columns; default *False*
        raw (False) : bool
            the columns will contain the raw values
        ignore_value2text_conversions (False) : bool
            valid only for the channels that have value to text conversions and
            if *raw=False*. If this is True then the raw numeric values will be
            used, and the conversion will not be applied.
        use_interpolation (True) : bool
            option to perform interpoaltions when multiple timestamp raster are
            present. If *False* then dataframe columns will be automatically
            filled with NaN's were the dataframe index values are not found in
            the current column's timestamps
        only_basenames (False) : bool
            use jsut the field names, without prefix, for structures and channel
            arrays
        interpolate_outwards_with_nan : bool
            use NaN values for the samples that lie outside of the original
            signal's timestamps
        chunk_ram_size : int
            desired data frame RAM usage in bytes; default 200 MB


        Returns
        -------
        dataframe : pandas.DataFrame
            yields pandas DataFrame's that should not exceed 200MB of RAM

        """

        self._link_attributes()

        if channels:
            mdf = self.filter(channels)

            result = mdf.iter_to_dataframe(
                raster=raster,
                time_from_zero=time_from_zero,
                empty_channels=empty_channels,
                keep_arrays=keep_arrays,
                use_display_names=use_display_names,
                time_as_date=time_as_date,
                reduce_memory_usage=reduce_memory_usage,
                raw=raw,
                ignore_value2text_conversions=ignore_value2text_conversions,
                use_interpolation=use_interpolation,
                only_basenames=only_basenames,
                chunk_ram_size=chunk_ram_size,
            )

            for df in result:
                yield df

            mdf.close()

        df = pd.DataFrame()
        self._set_temporary_master(None)

        if raster:
            try:
                raster = float(raster)
                assert raster > 0
            except (TypeError, ValueError):
                if isinstance(raster, str):
                    raster = self.get(raster, raw=True, ignore_invalidation_bits=True).timestamps
                else:
                    raster = np.array(raster)
            else:
                raster = master_using_raster(self, raster)
            master = raster
        else:
            masters = {index: self.get_master(index) for index in self.virtual_groups}

            if masters:
                master = reduce(np.union1d, masters.values())
            else:
                master = np.array([], dtype="<f4")

        master_ = master
        channel_count = sum(len(gp.channels) - 1 for gp in self.groups) + 1
        # approximation with all float64 dtype
        itemsize = channel_count * 8
        # use 200MB DataFrame chunks
        chunk_count = chunk_ram_size // itemsize or 1

        chunks, r = divmod(len(master), chunk_count)
        if r:
            chunks += 1

        for i in range(chunks):

            master = master_[chunk_count * i : chunk_count * (i + 1)]
            start = master[0]
            end = master[-1]

            df = pd.DataFrame()
            self._set_temporary_master(None)

            df["timestamps"] = pd.Series(master, index=np.arange(len(master)))
            df.set_index("timestamps", inplace=True)

            used_names = UniqueDB()
            used_names.get_unique_name("timestamps")

            groups_nr = len(self.virtual_groups)

            for group_index, virtual_group in self.virtual_groups.items():
                group_cycles = virtual_group.cycles_nr
                if group_cycles == 0 and empty_channels == "skip":
                    continue

                record_offset = max(
                    np.searchsorted(masters[group_index], start).flatten()[0] - 1, 0
                )
                stop = np.searchsorted(masters[group_index], end).flatten()[0]
                record_count = min(stop - record_offset + 1, group_cycles)

                channels = [
                    (None, gp_index, ch_index)
                    for gp_index, channel_indexes in self.included_channels(group_index)[group_index].items()
                    for ch_index in channel_indexes
                ]
                signals = [
                    signal.validate(copy=False)
                    for signal in self.select(
                        channels,
                        raw=True,
                        copy_master=False,
                        record_offset=record_offset,
                        record_count=record_count,
                        validate=True,
                    )
                ]

                if not signals:
                    continue

                for sig in signals:
                    if len(sig) == 0:
                        if empty_channels == "zeros":
                            sig.samples = np.zeros(
                                len(df.index), dtype=sig.samples.dtype
                            )
                            sig.timestamps = master
                        else:
                            continue

                if not raw:
                    if ignore_value2text_conversions:
                        if self.version < "4.00":
                            text_conversion = 11
                        else:
                            text_conversion = 7

                        for signal in signals:
                            conversion = signal.conversion
                            if (
                                conversion
                                and conversion.conversion_type < text_conversion
                            ):
                                signal.samples = conversion.convert(signal.samples)

                    else:
                        for signal in signals:
                            if signal.conversion:
                                signal.samples = signal.conversion.convert(
                                    signal.samples
                                )

                if use_interpolation and not np.array_equal(
                    master, signals[0].timestamps
                ):

                    if interpolate_outwards_with_nan:
                        timestamps = signals[0].timestamps
                        idx = np.argwhere(
                            (master >= timestamps[0]) & (master <= timestamps[-1])
                        ).flatten()

                    signals = [
                        signal.interp(master, self._integer_interpolation)
                        for signal in signals
                    ]

                    if interpolate_outwards_with_nan:
                        for sig in signals:
                            sig.timestamps = sig.timestamps[idx]
                            sig.samples = sig.samples[idx]

                signals = [sig for sig in signals if len(sig)]

                for k, sig in enumerate(signals):
                    # byte arrays
                    if len(sig.samples.shape) > 1:

                        if use_display_names:
                            channel_name = sig.display_name or sig.name
                        else:
                            channel_name = sig.name

                        channel_name = used_names.get_unique_name(channel_name)

                        df[channel_name] = pd.Series(
                            list(sig.samples), index=sig.timestamps,
                        )

                    # arrays and structures
                    elif sig.samples.dtype.names:
                        for name, series in components(
                            sig.samples,
                            sig.name,
                            used_names,
                            master=sig.timestamps,
                            only_basenames=only_basenames,
                        ):
                            df[name] = series

                    # scalars
                    else:
                        if use_display_names:
                            channel_name = sig.display_name or sig.name
                        else:
                            channel_name = sig.name

                        channel_name = used_names.get_unique_name(channel_name)

                        if reduce_memory_usage and sig.samples.dtype.kind in "SU":
                            unique = np.unique(sig.samples)
                            if len(sig.samples) / len(unique) >= 2:
                                df[channel_name] = pd.Series(
                                    sig.samples, index=sig.timestamps, dtype="category"
                                )
                            else:
                                df[channel_name] = pd.Series(
                                    sig.samples, index=sig.timestamps
                                )
                        else:
                            if reduce_memory_usage:
                                sig.samples = downcast(sig.samples)
                            df[channel_name] = pd.Series(
                                sig.samples, index=sig.timestamps
                            )

                if self._callback:
                    self._callback(group_index + 1, groups_nr)

            if time_as_date:
                new_index = np.array(df.index) + self.header.start_time.timestamp()
                new_index = pd.to_datetime(new_index, unit="s")

                df.set_index(new_index, inplace=True)
            elif time_from_zero and len(master):
                df.set_index(df.index - df.index[0], inplace=True)

            yield df

    def to_dataframe(
        self,
        channels=None,
        raster=None,
        time_from_zero=True,
        empty_channels="skip",
        keep_arrays=False,
        use_display_names=False,
        time_as_date=False,
        reduce_memory_usage=False,
        raw=False,
        ignore_value2text_conversions=False,
        use_interpolation=True,
        only_basenames=False,
        interpolate_outwards_with_nan=False,
    ):
        """ generate pandas DataFrame

        Parameters
        ----------
        channels : list
            filter a subset of channels; default *None*
        raster : float | np.array | str
            new raster that can be

            * a float step value
            * a channel name who's timestamps will be used as raster (starting with asammdf 5.5.0)
            * an array (starting with asammdf 5.5.0)

            see `resample` for examples of urisng this argument

        time_from_zero : bool
            adjust time channel to start from 0; default *True*
        empty_channels : str
            behaviour for channels without samples; the options are *skip* or
            *zeros*; default is *skip*
        use_display_names : bool
            use display name instead of standard channel name, if available.
        keep_arrays : bool
            keep arrays and structure channels as well as the
            component channels. If *True* this can be very slow. If *False*
            only the component channels are saved, and their names will be
            prefixed with the parent channel.
        time_as_date : bool
            the dataframe index will contain the datetime timestamps
            according to the measurement start time; default *False*. If
            *True* then the argument ``time_from_zero`` will be ignored.
        reduce_memory_usage : bool
            reduce memory usage by converting all float columns to float32 and
            searching for minimum dtype that can reprezent the values found
            in integer columns; default *False*
        raw (False) : bool
            the columns will contain the raw values

            .. versionadded:: 5.7.0

        ignore_value2text_conversions (False) : bool
            valid only for the channels that have value to text conversions and
            if *raw=False*. If this is True then the raw numeric values will be
            used, and the conversion will not be applied.

            .. versionadded:: 5.8.0

        use_interpolation (True) : bool
            option to perform interpoaltions when multiple timestamp raster are
            present. If *False* then dataframe columns will be automatically
            filled with NaN's were the dataframe index values are not found in
            the current column's timestamps

            .. versionadded:: 5.11.0

        only_basenames (False) : bool
            use jsut the field names, without prefix, for structures and channel
            arrays

            .. versionadded:: 5.13.0

        interpolate_outwards_with_nan : bool
            use NaN values for the samples that lie outside of the original
            signal's timestamps

            .. versionadded:: 5.15.0

        Returns
        -------
        dataframe : pandas.DataFrame

        """

        self._link_attributes()

        if channels:
            mdf = self.filter(channels)

            result = mdf.to_dataframe(
                raster=raster,
                time_from_zero=time_from_zero,
                empty_channels=empty_channels,
                keep_arrays=keep_arrays,
                use_display_names=use_display_names,
                time_as_date=time_as_date,
                reduce_memory_usage=reduce_memory_usage,
                raw=raw,
                ignore_value2text_conversions=ignore_value2text_conversions,
                use_interpolation=use_interpolation,
                only_basenames=only_basenames,
                interpolate_outwards_with_nan=interpolate_outwards_with_nan,
            )

            mdf.close()
            return result

        df = pd.DataFrame()
        self._set_temporary_master(None)

        if raster:
            try:
                raster = float(raster)
                assert raster > 0
            except (TypeError, ValueError):
                if isinstance(raster, str):
                    raster = self.get(raster).timestamps
                else:
                    raster = np.array(raster)
            else:
                raster = master_using_raster(self, raster)
            master = raster
        else:
            masters = {index: self.get_master(index) for index in self.virtual_groups}

            if masters:
                master = reduce(np.union1d, masters.values())
            else:
                master = np.array([], dtype="<f4")

            del masters

        df["timestamps"] = pd.Series(master, index=np.arange(len(master)))
        df.set_index("timestamps", inplace=True)

        used_names = UniqueDB()
        used_names.get_unique_name("timestamps")

        groups_nr = len(self.virtual_groups)

        for group_index, (virtual_group_index, virtual_group) in enumerate(self.virtual_groups.items()):
            if virtual_group.cycles_nr == 0 and empty_channels == "skip":
                continue

            channels = [
                (None, gp_index, ch_index)
                for gp_index, channel_indexes in self.included_channels(virtual_group_index)[virtual_group_index].items()
                for ch_index in channel_indexes
                if ch_index != self.masters_db.get(gp_index, None)
            ]

            signals = [
                signal.validate(copy=False)
                for signal in self.select(
                    channels,
                    raw=True,
                    copy_master=False,
                    validate=True,
                )
            ]

            if not signals:
                continue

            for sig in signals:
                if len(sig) == 0:
                    if empty_channels == "zeros":
                        sig.samples = np.zeros(len(df.index), dtype=sig.samples.dtype)
                        sig.timestamps = master
                    else:
                        continue

            if not raw:
                if ignore_value2text_conversions:

                    for signal in signals:
                        conversion = signal.conversion
                        if conversion:
                            samples = conversion.convert(signal.samples)
                            if samples.dtype.kind not in 'US':
                                signal.samples = samples
                else:
                    for signal in signals:
                        if signal.conversion:
                            signal.samples = signal.conversion.convert(signal.samples)

            if use_interpolation and not np.array_equal(master, signals[0].timestamps):

                if interpolate_outwards_with_nan:
                    timestamps = signals[0].timestamps
                    idx = np.argwhere(
                        (master >= timestamps[0]) & (master <= timestamps[-1])
                    ).flatten()

                signals = [
                    signal.interp(master, self._integer_interpolation)
                    for signal in signals
                ]

                if interpolate_outwards_with_nan:
                    for sig in signals:
                        sig.timestamps = sig.timestamps[idx]
                        sig.samples = sig.samples[idx]

            signals = [sig for sig in signals if len(sig)]

            for k, sig in enumerate(signals):
                # byte arrays
                if len(sig.samples.shape) > 1:

                    if use_display_names:
                        channel_name = sig.display_name or sig.name
                    else:
                        channel_name = sig.name

                    channel_name = used_names.get_unique_name(channel_name)

                    df[channel_name] = pd.Series(
                        list(sig.samples), index=sig.timestamps,
                    )

                # arrays and structures
                elif sig.samples.dtype.names:
                    for name, series in components(
                        sig.samples,
                        sig.name,
                        used_names,
                        master=sig.timestamps,
                        only_basenames=only_basenames,
                    ):
                        df[name] = series

                # scalars
                else:
                    if use_display_names:
                        channel_name = sig.display_name or sig.name
                    else:
                        channel_name = sig.name

                    channel_name = used_names.get_unique_name(channel_name)

                    if reduce_memory_usage and sig.samples.dtype.kind in "SU":
                        unique = np.unique(sig.samples)
                        if len(sig.samples) / len(unique) >= 2:
                            df[channel_name] = pd.Series(
                                sig.samples, index=sig.timestamps, dtype="category"
                            )
                        else:
                            df[channel_name] = pd.Series(
                                sig.samples, index=sig.timestamps
                            )
                    else:
                        if reduce_memory_usage:
                            sig.samples = downcast(sig.samples)
                        df[channel_name] = pd.Series(sig.samples, index=sig.timestamps)

            if self._callback:
                self._callback(group_index + 1, groups_nr)

        if time_as_date:
            new_index = np.array(df.index) + self.header.start_time.timestamp()
            new_index = pd.to_datetime(new_index, unit="s")

            df.set_index(new_index, inplace=True)
        elif time_from_zero and len(master):
            df.set_index(df.index - df.index[0], inplace=True)

        return df

    def extract_can_logging(
        self, dbc_files, version=None, ignore_invalid_signals=False
    ):
        """ extract all possible CAN signal using the provided databases.

        Parameters
        ----------
        dbc_files : iterable
            iterable of str or pathlib.Path objects
        version (None) : str
            output file version
        ignore_invalid_signals (False) : bool
            ignore signals that have all samples equal to their maximum value

            .. versionadded:: 5.7.0

        Returns
        -------
        mdf : MDF
            new MDF file that contains the succesfully extracted signals

        """
        self._link_attributes()

        if version is None:
            version = self.version
        else:
            version = validate_version_argument(version)

        out = MDF(version=version, callback=self._callback)
        out.header.start_time = self.header.start_time

        if self._callback:
            out._callback = out._mdf._callback = self._callback

        max_flags = []

        valid_dbc_files = []
        for dbc_name in dbc_files:
            dbc = load_can_database(dbc_name)
            if dbc is None:
                continue
            else:
                valid_dbc_files.append((dbc, dbc_name))

        count = sum(1 for group in self.groups if group.CAN_logging)
        count *= len(valid_dbc_files)

        cntr = 0

        total_unique_ids = set()
        found_ids = defaultdict(set)
        not_found_ids = defaultdict(list)
        unknown_ids = defaultdict(list)

        for dbc, dbc_name in valid_dbc_files:
            is_j1939 = dbc.contains_j1939
            if is_j1939:
                messages = {message.arbitration_id.pgn: message for message in dbc}
            else:
                messages = {message.arbitration_id.id: message for message in dbc}

            current_not_found_ids = {
                (msg_id, message.name) for msg_id, message in messages.items()
            }

            msg_map = {}

            for i, group in enumerate(self.groups):
                if not group.CAN_logging:
                    continue

                if not "CAN_DataFrame" in [ch.name for ch in group.channels]:
                    continue

                parents, dtypes = self._prepare_record(group)
                data = self._load_data(group, optimize_read=False)

                for fragment_index, fragment in enumerate(data):
                    if dtypes.itemsize:
                        group.record = np.core.records.fromstring(
                            fragment[0], dtype=dtypes
                        )
                    else:
                        group.record = None
                        continue

                    self._set_temporary_master(None)
                    self._set_temporary_master(self.get_master(i, data=fragment))

                    bus_ids = self.get(
                        "CAN_DataFrame.BusChannel",
                        group=i,
                        data=fragment,
                        samples_only=True,
                    )[0].astype("<u1")

                    msg_ids = (
                        self.get("CAN_DataFrame.ID", group=i, data=fragment,)
                        & 0x1FFFFFFF
                    )

                    if is_j1939:
                        ps = (msg_ids.samples >> 8) & 0xFF
                        pf = (msg_ids.samples >> 16) & 0xFF
                        _pgn = pf << 8
                        msg_ids.samples = np.where(pf >= 240, _pgn + ps, _pgn,)

                    data_bytes = self.get(
                        "CAN_DataFrame.DataBytes",
                        group=i,
                        data=fragment,
                        samples_only=True,
                    )[0]

                    buses = np.unique(bus_ids)

                    for bus in buses:
                        idx = np.argwhere(bus_ids == bus).ravel()
                        bus_t = msg_ids.timestamps[idx]
                        bus_msg_ids = msg_ids.samples[idx]
                        bus_data_bytes = data_bytes[idx]

                        unique_ids = np.unique(bus_msg_ids).astype("<u8")

                        total_unique_ids = total_unique_ids | set(unique_ids)

                        for msg_id in unique_ids:
                            message = messages.get(msg_id, None)
                            if message is None:
                                unknown_ids[msg_id].append(True)
                                continue

                            found_ids[dbc_name].add((msg_id, message.name))
                            try:
                                current_not_found_ids.remove((msg_id, message.name))
                            except KeyError:
                                pass

                            unknown_ids[msg_id].append(False)

                            idx = np.argwhere(bus_msg_ids == msg_id).ravel()
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

                                    index = len(out.groups)
                                    msg_map[entry] = index

                                    for name_, signal in signals.items():
                                        sig = Signal(
                                            samples=signal["samples"],
                                            timestamps=signal["t"],
                                            name=signal["name"],
                                            comment=signal["comment"],
                                            unit=signal["unit"],
                                            invalidation_bits=signal["invalidation_bits"] if ignore_invalid_signals else None,
                                        )

                                        sig.comment = f"""\
<CNcomment>
<TX>{sig.comment}</TX>
<names>
    <display>
        CAN{bus}.{message.name}.{signal['name']}
    </display>
</names>
</CNcomment>"""
                                        sigs.append(sig)

                                    cg_nr = out.append(
                                        sigs,
                                        f"from CAN{bus} message ID=0x{msg_id:X}",
                                        common_timebase=True,
                                    )

                                    out.groups[
                                        cg_nr
                                    ].channel_group.comment = f"{message} 0x{msg_id:X}"

                                    if ignore_invalid_signals:
                                        max_flags.append([False])
                                        for ch_index, sig in enumerate(sigs, 1):
                                            max_flags[cg_nr].append(np.all(sig.invalidation_bits))

                                else:

                                    index = msg_map[entry]

                                    sigs = []

                                    for name_, signal in signals.items():

                                        sigs.append((signal["samples"], signal["invalidation_bits"] if ignore_invalid_signals else None))

                                        t = signal["t"]

                                    if ignore_invalid_signals:
                                        for ch_index, sig in enumerate(sigs, 1):
                                            max_flags[index][ch_index] = max_flags[index][ch_index] or np.all(sig[1])

                                    sigs.insert(0, (t, None))

                                    out.extend(index, sigs)
                    self._set_temporary_master(None)
                    group.record = None
                cntr += 1
                if self._callback:
                    self._callback(cntr, count)

            if current_not_found_ids:
                not_found_ids[dbc_name] = list(current_not_found_ids)

        unknown_ids = {
            msg_id for msg_id, not_found in unknown_ids.items() if all(not_found)
        }

        self.last_call_info = {
            "dbc_files": dbc_files,
            "total_unique_ids": total_unique_ids,
            "unknown_id_count": len(unknown_ids),
            "not_found_ids": not_found_ids,
            "found_ids": found_ids,
            "unknown_ids": unknown_ids,
        }

        if ignore_invalid_signals:
            to_keep = []

            for i, group in enumerate(out.groups):
                for j, channel in enumerate(group.channels[1:], 1):
                    if not max_flags[i][j-1]:
                        to_keep.append((None, i, j))

            tmp = out.filter(to_keep, version)
            out.close()
            out = tmp

        if self._callback:
            self._callback(100, 100)
        if not out.groups:
            logger.warning(
                f'No CAN signals could be extracted from "{self.name}". The'
                "output file will be empty."
            )

        return out

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

        self._mdf.configure(
            read_fragment_size=read_fragment_size,
            write_fragment_size=write_fragment_size,
            use_display_names=use_display_names,
            single_bit_uint_as_bool=single_bit_uint_as_bool,
            integer_interpolation=integer_interpolation,
            copy_on_get=copy_on_get,
        )
        self._link_attributes()


if __name__ == "__main__":
    pass

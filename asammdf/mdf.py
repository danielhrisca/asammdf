# -*- coding: utf-8 -*-
""" common MDF file format module """

import csv
import os
import sys
from warnings import warn
from functools import reduce
from struct import unpack

import numpy as np
from pandas import DataFrame

from .mdf_v2 import MDF2
from .mdf_v3 import MDF3
from .mdf_v4 import MDF4
from .utils import MdfException, get_text_v3, get_text_v4
from .v2_v3_blocks import TextBlock as TextBlockV3
from .v2_v3_blocks import Channel as ChannelV3
from .v4_blocks import Channel as ChannelV4
from .v4_blocks import TextBlock as TextBlockV4
from .v4_blocks import ChannelArrayBlock

PYVERSION = sys.version_info[0]

MDF2_VERSIONS = ('2.00', '2.10', '2.14')
MDF3_VERSIONS = ('3.00', '3.10', '3.20', '3.30')
MDF4_VERSIONS = ('4.00', '4.10', '4.11')
SUPPORTED_VERSIONS = MDF2_VERSIONS + MDF3_VERSIONS + MDF4_VERSIONS

__all__ = ['MDF', 'SUPPORTED_VERSIONS']


class MDF(object):
    """Unified access to MDF v3 and v4 files.

    Parameters
    ----------
    name : string
        mdf file name, if provided it must be a real file name
    memory : str
        memory option; default `full`

            * if *full* the data group binary data block will be loaded in RAM
            * if *low* the channel data is read from disk on request, and the
                metadata is loaded into RAM

            * if *minimum* only minimal data is loaded into RAM

    version : string
        mdf file version from ('2.00', '2.10', '2.14', '3.00', '3.10', '3.20',
        '3.30', '4.00', '4.10', '4.11'); default '4.10'

    """

    _iter_channels = True

    def __init__(self, name=None, memory='full', version='4.10'):
        if name:
            if os.path.isfile(name):
                with open(name, 'rb') as file_stream:
                    file_stream.read(8)
                    version = file_stream.read(4).decode('ascii').strip(' \0')
                    if not version:
                        file_stream.read(16)
                        version = unpack('<H', file_stream.read(2))[0]
                        version = str(version)
                        version = '{}.{}'.format(version[0], version[1:])
                if version in MDF3_VERSIONS:
                    self._mdf = MDF3(name, memory)
                elif version in MDF4_VERSIONS:
                    self._mdf = MDF4(name, memory)
                elif version in MDF2_VERSIONS:
                    self._mdf = MDF2(name, memory)
                else:
                    message = ('"{}" is not a supported MDF file; '
                               '"{}" file version was found')
                    raise MdfException(message.format(name, version))
            else:
                raise MdfException('File "{}" does not exist'.format(name))
        else:
            if version in MDF2_VERSIONS:
                self._mdf = MDF3(
                    version=version,
                    memory=memory,
                )
            elif version in MDF3_VERSIONS:
                self._mdf = MDF3(
                    version=version,
                    memory=memory,
                )
            elif version in MDF4_VERSIONS:
                self._mdf = MDF4(
                    version=version,
                    memory=memory,
                )

        # link underlying _file attributes and methods to the new MDF object
        for attr in set(dir(self._mdf)) - set(dir(self)):
            setattr(self, attr, getattr(self._mdf, attr))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _excluded_channels(self, index):
        group = self.groups[index]

        excluded_channels = set()
        master_index = self.masters_db.get(index, -1)
        excluded_channels.add(master_index)

        channels = group['channels']

        if self.version in MDF2_VERSIONS + MDF3_VERSIONS:
            for dep in group['channel_dependencies']:
                if dep is None:
                    continue
                for ch_nr, gp_nr in dep.referenced_channels:
                    if gp_nr == index:
                        excluded_channels.add(ch_nr)
        else:
            for dependencies in group['channel_dependencies']:
                if dependencies is None:
                    continue
                if all(not isinstance(dep, ChannelArrayBlock)
                       for dep in dependencies):
                    for channel in dependencies:
                        excluded_channels.add(channels.index(channel))
                else:
                    for dep in dependencies:
                        for ch_nr, gp_nr in dep.referenced_channels:
                            if gp_nr == index:
                                excluded_channels.add(ch_nr)

        return excluded_channels

    def __contains__(self, channel):
        return channel in self.channels_db

    def __iter__(self):
        # the default is to yield from iter_channels
        for signal in self.iter_channels():
            yield signal

    def convert(self, to, memory='full'):
        """convert MDF to other versions

        Parameters
        ----------
        to : str
            new mdf file version from ('2.00', '2.10', '2.14', '3.00', '3.10',
            '3.20', '3.30', '4.00', '4.10', '4.11'); default '4.10'
        memory : str
            memory option; default `full`

        Returns
        -------
        out : MDF
            new MDF object

        """
        if to not in SUPPORTED_VERSIONS:
            message = (
                'Unknown output mdf version "{}".'
                ' Available versions are {}'
            )
            warn(message.format(to, SUPPORTED_VERSIONS))
            out = None
        else:
            out = MDF(version=to, memory=memory)

            # walk through all groups and get all channels
            for i, group in enumerate(self.groups):
                sigs = []
                excluded_channels = self._excluded_channels(i)

                data = self._load_group_data(group)

                for j, _ in enumerate(group['channels']):
                    if j in excluded_channels:
                        continue
                    else:
                        sigs.append(self.get(group=i, index=j, data=data))

                if sigs:
                    source_info = 'Converted from {} to {}'
                    out.append(
                        sigs,
                        source_info.format(self.version, to),
                        common_timebase=True,
                    )

        return out

    def cut(self, start=None, stop=None, whence=0):
        """convert MDF to other versions

        Parameters
        ----------
        start : float
            start time, default None. If *None* then the start of measurement
            is used
        stop : float
            stop time, default . If *None* then the end of measurement is used
        whence : int
            how to search for the start and stop values

            * 0 : absolute
            * 1 : relative to first timestamp

        Returns
        -------
        out : MDF
            new MDF object

        """
        out = MDF(
            version=self.version,
            memory=self.memory,
        )

        if whence == 1:
            timestamps = []
            for i, _ in enumerate(self.groups):
                master_index = self.masters_db.get(i, None)
                if master_index is not None:
                    master = self.get(
                        group=i,
                        index=master_index,
                        samples_only=True,
                    )
                    if master.size:
                        timestamps.append(master[0])
            first_timestamp = np.amin(timestamps)
            if start is not None:
                start += first_timestamp
            if stop is not None:
                stop += first_timestamp

            timestamps = None
            del timestamps

        # walk through all groups and get all channels
        for i, group in enumerate(self.groups):
            sigs = []
            excluded_channels = self._excluded_channels(i)

            data = self._load_group_data(group)

            for j, _ in enumerate(group['channels']):
                if j in excluded_channels:
                    continue
                sig = self.get(
                    group=i,
                    index=j,
                    data=data
                ).cut(start=start, stop=stop)
                sigs.append(sig)

            data = None
            del data

            if sigs:
                if start:
                    start_ = '{}s'.format(start)
                else:
                    start_ = 'start of measurement'
                if stop:
                    stop_ = '{}s'.format(stop)
                else:
                    stop_ = 'end of measurement'
                out.append(
                    sigs,
                    'Cut from {} to {}'.format(start_, stop_),
                    common_timebase=True,
                )
        return out

    def export(self, fmt, filename=None):
        """ export MDF to other formats. The *MDF* file name is used is
        available, else the *filename* aragument must be provided.

        Parameters
        ----------
        fmt : string
            can be one of the following:

                * `csv` : CSV export that uses the ";" delimiter. This option
                    will generate a new csv file for each data group
                    (<MDFNAME>_DataGroup_<cntr>.csv)

                * `hdf5` : HDF5 file output; each *MDF* data group is mapped to
                    a *HDF5* group with the name 'DataGroup_<cntr>'
                    (where <cntr> is the index)

                * `excel` : Excel file output (very slow). This option will
                    generate a new excel file for each data group
                    (<MDFNAME>_DataGroup_<cntr>.xlsx)

                * `mat` : Matlab .mat version 5 export, for Matlab >= 7.6. In
                    the mat file the channels will be renamed to
                    'DataGroup_<cntr>_<channel name>'. The channel group master
                    will be renamed to 'DataGroup_<cntr>_<channel name>_master'
                    ( *<cntr>* is the data group index starting from 0)

        filename : string
            export file name

        """

        header_items = (
            'date',
            'time',
            'author',
            'organization',
            'project',
            'subject',
        )

        if filename is None and self.name is None:
            message = ('Must specify filename for export'
                       'if MDF was created without a file name')
            warn(message)
            return

        name = filename if filename else self.name
        if fmt == 'hdf5':
            try:
                from h5py import File as HDF5
            except ImportError:
                warn('h5py not found; export to HDF5 is unavailable')
                return
            else:
                if not name.endswith('.hdf'):
                    name = os.path.splitext(name)[0] + '.hdf'
                with HDF5(name, 'w') as hdf:
                    # header information
                    group = hdf.create_group(os.path.basename(name))

                    if self.version in MDF2_VERSIONS + MDF3_VERSIONS:
                        for item in header_items:
                            group.attrs[item] = self.header[item]

                    # save each data group in a HDF5 group called
                    # "DataGroup_<cntr>" with the index starting from 1
                    # each HDF5 group will have a string attribute "master"
                    # that will hold the name of the master channel
                    for i, grp in enumerate(self.groups):
                        group_name = r'/' + 'DataGroup_{}'.format(i + 1)
                        group = hdf.create_group(group_name)

                        master_index = self.masters_db.get(i, -1)

                        data = self._load_group_data(grp)

                        for j, _ in enumerate(grp['channels']):
                            sig = self.get(group=i, index=j, data=data)
                            name = sig.name
                            if j == master_index:
                                group.attrs['master'] = name
                            dataset = group.create_dataset(name,
                                                           data=sig.samples)
                            if sig.unit:
                                dataset.attrs['unit'] = sig.unit
                            if sig.comment:
                                dataset.attrs['comment'] = sig.comment

        elif fmt == 'excel':
            try:
                import xlsxwriter
            except ImportError:
                warn('xlsxwriter not found; export to Excel unavailable')
                return
            else:
                excel_name = os.path.splitext(name)[0]
                count = len(self.groups)
                for i, grp in enumerate(self.groups):
                    print('Exporting group {} of {}'.format(i + 1, count))

                    data = self._load_group_data(grp)

                    group_name = 'DataGroup_{}'.format(i + 1)
                    wb_name = '{}_{}.xlsx'.format(excel_name, group_name)
                    workbook = xlsxwriter.Workbook(wb_name)
                    bold = workbook.add_format({'bold': True})

                    if self.version in MDF2_VERSIONS + MDF3_VERSIONS:
                        sheet = workbook.add_worksheet(group_name)
                        for j, item in enumerate(header_items):
                            sheet.write(j, 0, item.title(), bold)
                            sheet.write(j, 1, self.header[item].decode('latin-1'))

                        # the sheet header has 3 rows
                        # the channel name and unit 'YY [xx]'
                        # the channel comment
                        # the flag for data grup master channel
                        sheet.write(0, 0, 'Channel', bold)
                        sheet.write(1, 0, 'comment', bold)
                        sheet.write(2, 0, 'is master', bold)

                        master_index = self.masters_db.get(i, -1)

                        for j in range(grp['channel_group']['cycles_nr']):
                            sheet.write(j + 3, 0, str(j))

                        for j, _ in enumerate(grp['channels']):
                            sig = self.get(group=i, index=j, data=data)

                            col = j + 1
                            sig_description = '{} [{}]'.format(
                                sig.name,
                                sig.unit,
                            )
                            comment = sig.comment if sig.comment else ''
                            sheet.write(0, col, sig_description)
                            sheet.write(1, col, comment)
                            if j == master_index:
                                sheet.write(2, col, 'x')
                            sheet.write_column(3, col, sig.samples.astype(str))

                    workbook.close()

        elif fmt == 'csv':
            csv_name = os.path.splitext(name)[0]
            count = len(self.groups)
            for i, grp in enumerate(self.groups):
                print('Exporting group {} of {}'.format(i + 1, count))
                data = self._load_group_data(grp)

                group_name = 'DataGroup_{}'.format(i + 1)
                group_csv_name = '{}_{}.csv'.format(csv_name, group_name)
                with open(group_csv_name, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=';')

                    ch_nr = len(grp['channels'])
                    channels = [
                        self.get(group=i, index=j, data=data)
                        for j in range(ch_nr)
                    ]

                    master_index = self.masters_db.get(i, -1)
                    cycles = grp['channel_group']['cycles_nr']

                    names_row = ['Channel', ]
                    names_row += [
                        '{} [{}]'.format(ch.name, ch.unit)
                        for ch in channels
                    ]
                    writer.writerow(names_row)

                    comment_row = ['comment', ]
                    comment_row += [ch.comment for ch in channels]
                    writer.writerow(comment_row)

                    master_row = ['Is master', ]
                    master_row += [
                        'x' if j == master_index else ''
                        for j in range(ch_nr)
                    ]
                    writer.writerow(master_row)

                    vals = [np.array(range(cycles), dtype=np.uint32), ]
                    vals += [ch.samples for ch in channels]

                    writer.writerows(zip(*vals))

        elif fmt == 'mat':
            try:
                from scipy.io import savemat
            except ImportError:
                warn('scipy not found; export to mat is unavailable')
                return

            name = os.path.splitext(name)[0] + '.mat'
            mdict = {}

            master = 'DataGroup_{}_{}_master'
            channel = 'DataGroup_{}_{}'

            for i, grp in enumerate(self.groups):
                master_index = self.masters_db.get(i, -1)
                data = self._load_group_data(grp)
                for j, _ in enumerate(grp['channels']):
                    sig = self.get(
                        group=i,
                        index=j,
                        data=data,
                    )
                    if j == master_index:
                        channel_name = master.format(i, sig.name)
                    else:
                        channel_name = channel.format(i, sig.name)
                    mdict[channel_name] = sig.samples

            savemat(
                name,
                mdict,
                long_field_names=True,
                do_compression=True,
            )
        else:
            message = (
                'Unsopported export type "{}". '
                'Please select "csv", "excel", "hdf5" or "mat"'
            )
            warn(message.format(fmt))

    def filter(self, channels, memory=None):
        """ return new *MDF* object that contains only the channels listed in
        *channels* argument

        Parameters
        ----------
        channels : list
            list of items to be filtered; each item can be :

                * a channel name string
                * (channel_name, group index, channel index) list or tuple
                * (channel name, group index) list or tuple
                * (None, group index, channel index) lsit or tuple

        memory : str
            memory option for filtered mdf; default None in which case the
            original file's memory option is used

        Returns
        -------
        mdf : MDF
            new MDF file

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

        # group channels by group index
        gps = {}

        for item in channels:
            if isinstance(item, (list, tuple)):
                if len(item) not in (2, 3):
                    raise MdfException(
                        'The items used for filtering must be strings, '
                        'or they must match the first 3 argumens of the get '
                        'method'
                    )
                else:
                    group, index = self._validate_channel_selection(*item)
                    if group not in gps:
                        gps[group] = set()
                    gps[group].add(index)
            else:
                name = item
                group, index = self._validate_channel_selection(name)
                if group not in gps:
                    gps[group] = set()
                gps[group].add(index)

        for group_index, indexes in gps.items():
            grp = self.groups[group_index]
            excluded_channels = set()
            for index in indexes:
                if self.version in MDF2_VERSIONS + MDF3_VERSIONS:
                    dep = grp['channel_dependencies'][index]
                    if dep:
                        for ch_nr, gp_nr in dep.referenced_channels:
                            if gp_nr == group:
                                excluded_channels.add(ch_nr)
                else:
                    dependencies = grp['channel_dependencies'][index]
                    if dependencies is None:
                        continue
                    if all(not isinstance(dep, ChannelArrayBlock)
                           for dep in dependencies):
                        channels = grp['channels']
                        for channel in dependencies:
                            excluded_channels.add(channels.index(channel))
                    else:
                        for dep in dependencies:
                            for ch_nr, gp_nr in dep.referenced_channels:
                                if gp_nr == group:
                                    excluded_channels.add(ch_nr)

            gps[group_index] = gps[group_index] - excluded_channels

        if memory is None or memory not in ('full', 'low', 'minimum'):
            memory = self.memory

        mdf = MDF(
            version=self.version,
            memory=memory,
        )

        # append filtered channels to new MDF
        for group in gps:
            grp = self.groups[group]
            data = self._load_group_data(grp)
            sigs = []
            for index in gps[group]:
                sigs.append(self.get(group=group, index=index, data=data))
            if sigs:
                if self.name:
                    origin = os.path.basename(self.name)
                else:
                    origin = 'New MDF'
                source = 'Signals filtered from <{}>'.format(origin)
                mdf.append(
                    sigs,
                    source,
                    common_timebase=True,
                )

        return mdf

    @staticmethod
    def merge(files, outversion='4.10', memory='full'):
        """ merge several files and return the merged MDF object. The files
        must have the same internal structure (same number of groups, and same
        channels in each group)

        Parameters
        ----------
        files : list | tuple
            list of MDF file names or MDF instances
        outversion : str
            merged file version
        memory : str
            memory option; default `full`

        Returns
        -------
        merged : MDF
            new MDF object with merged channels

        Raises
        ------
        MdfException : if there are inconsistencies between the files
            merged MDF object
        """
        if not files:
            raise MdfException('No files given for merge')

        files = [
            file if isinstance(file, MDF) else MDF(file, memory)
            for file in files
        ]

        if not len(set(len(file.groups) for file in files)) == 1:
            message = (
                "Can't merge files: "
                "difference in number of data groups"
            )
            raise MdfException(message)

        merged = MDF(
            version=outversion,
            memory=memory,
        )

        for i, groups in enumerate(zip(*(file.groups for file in files))):
            channels_nr = set(len(group['channels']) for group in groups)
            if not len(channels_nr) == 1:
                message = (
                    "Can't merge files: "
                    "different channel number for data groups {}"
                )
                raise MdfException(message.format(i))

            signals = []
            mdf = files[0]
            excluded_channels = mdf._excluded_channels(i)

            groups_data = [
                files[index]._load_group_data(grp)
                for index, grp in enumerate(groups)
            ]

            group_channels = [group['channels'] for group in groups]
            for j, channels in enumerate(zip(*group_channels)):
                if memory == 'minimum':
                    names = []
                    for file in files:
                        if file.version in MDF2_VERSIONS + MDF3_VERSIONS:
                            grp = file.groups[i]
                            if grp['data_location'] == 0:
                                stream = file._file
                            else:
                                stream = file._tempfile

                            channel = ChannelV3(
                                address=grp['channels'][j],
                                stream=stream,
                            )

                            if channel.get('long_name_addr', 0):
                                name = get_text_v3(channel['long_name_addr'], stream)
                            else:
                                name = (
                                    channel['short_name']
                                    .decode('latin-1')
                                    .strip(' \r\n\t\0')
                                    .split('\\')[0]
                                )
                        else:
                            grp = file.groups[i]
                            if grp['data_location'] == 0:
                                stream = file._file
                            else:
                                stream = file._tempfile

                            channel = ChannelV4(
                                address=grp['channels'][j],
                                stream=stream,
                            )
                            name = get_text_v4(channel['name_addr'], stream)
                        name = name.split('\\')[0]
                        names.append(name)
                    names = set(names)
                else:
                    names = set(ch.name for ch in channels)
                if not len(names) == 1:
                    message = (
                        "Can't merge files: "
                        "different channel names for data group {}"
                    )
                    raise MdfException(message.format(i))

                if j in excluded_channels:
                    continue

                signals_to_merge = [
                    file.get(group=i, index=j, data=data)
                    for file, data in zip(files, groups_data)
                ]

                signal = signals_to_merge[0]
                for merged_signal in signals_to_merge[1:]:
                    signal = signal.extend(merged_signal)

                signals.append(signal)

            if signals:
                merged.append(signals, common_timebase=True)

        return merged

    def iter_channels(self, skip_master=True):
        """ generator that yields a `Signal` for each non-master channel

        Parameters
        ----------
        skip_master : bool
            do not yield master channels; default True

        """
        for i, group in enumerate(self.groups):
            try:
                master_index = self.masters_db[i]
            except KeyError:
                master_index = -1

            for j, _ in enumerate(group['channels']):
                if skip_master and j == master_index:
                    continue
                yield self.get(group=i, index=j)

    def iter_groups(self):
        """ generator that yields channel groups as pandas DataFrames"""

        for i, group in enumerate(self.groups):
            data = self._load_group_data(group)
            master_index = self.masters_db.get(i, None)
            if master_index is None:
                pandas_dict = {}
            else:
                master = self.get(
                    group=i,
                    index=master_index,
                    data=data,
                )
                pandas_dict = {master.name: master.samples}
            for j, _ in enumerate(group['channels']):
                if j == master_index:
                    continue
                sig = self.get(
                    group=i,
                    index=j,
                    data=data,
                )
                pandas_dict[sig.name] = sig.samples
            yield DataFrame.from_dict(pandas_dict)

    def resample(self, raster, memory=None):
        """ resample all channels to given raster

        Parameters
        ----------
        raster : float
            time raster is seconds
        memory : str
            memory option; default `None`

        Returns
        -------
        mdf : MDF
            new MDF with resampled channels

        """

        if memory is None or memory not in ('full', 'low', 'minimum'):
            memory = self.memory

        mdf = MDF(
            version=self.version,
            memory=memory,
        )

        # walk through all groups and get all channels
        for i, group in enumerate(self.groups):
            sigs = []
            excluded_channels = self._excluded_channels(i)

            data = self._load_group_data(group)

            for j, _ in enumerate(group['channels']):
                if j in excluded_channels:
                    continue
                sig = self.get(
                    group=i,
                    index=j,
                    data=data,
                    raster=raster,
                )
                sigs.append(sig)

            data = None
            del data

            if sigs:
                mdf.append(
                    sigs,
                    'Resampled to {}s'.format(raster),
                    common_timebase=True,
                )
        return mdf

    def select(self, channels, dataframe=False):
        """ return the channels listed in *channels* argument

        Parameters
        ----------
        channels : list
            list of items to be filtered; each item can be :

                * a channel name string
                * (channel_name, group index, channel index) list or tuple
                * (channel name, group index) list or tuple
                * (None, group index, channel index) lsit or tuple

        dataframe: bool
            return a pandas DataFrame instead of a list of Signals; in this
            case the signals will be interpolated using the union of all
            timestamps

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

        # group channels by group index
        gps = {}

        indexes = []

        for item in channels:
            if isinstance(item, (list, tuple)):
                if len(item) not in (2, 3):
                    raise MdfException(
                        'The items used for filtering must be strings, '
                        'or they must match the first 3 argumens of the get '
                        'method'
                    )
                else:
                    group, index = self._validate_channel_selection(*item)
                    indexes.append((group, index))
                    if group not in gps:
                        gps[group] = set()
                    gps[group].add(index)
            else:
                name = item
                group, index = self._validate_channel_selection(name)
                indexes.append((group, index))
                if group not in gps:
                    gps[group] = set()
                gps[group].add(index)

        signals = {}
        for group in gps:
            grp = self.groups[group]
            data = self._load_group_data(grp)
            for index in gps[group]:
                signal = self.get(group=group, index=index, data=data)
                signals[(group, index)] = signal

        signals = [signals[pair] for pair in indexes]

        if dataframe:
            times = [s.timestamps for s in signals]
            t = reduce(np.union1d, times).flatten().astype(np.float64)
            signals = [s.interp(t) for s in signals]
            times = None

            pandas_dict = {'t': t}
            for sig in signals:
                pandas_dict[sig.name] = sig.samples

            signals = DataFrame.from_dict(pandas_dict)

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
        if channel in self:
            return tuple(self.channels_db[channel])
        else:
            return tuple()


if __name__ == '__main__':
    pass

"""
MDF file format module

"""
import csv
import os
import warnings

import numpy as np

from .mdf3 import MDF3
from .mdf4 import MDF4
from .v3constants import CHANNEL_TYPE_MASTER as V3_MASTER
from .v4constants import CHANNEL_TYPE_MASTER as V4_MASTER
from .v4constants import CHANNEL_TYPE_VIRTUAL_MASTER as V4_VIRTUAL_MASTER


MDF3_VERSIONS = ('3.00', '3.10', '3.20', '3.30')
MDF4_VERSIONS = ('4.00', '4.10', '4.11')


__all__ = ['MDF', ]


class MDF(object):
    """Unified access to MDF v3 and v4 files.

    Parameters
    ----------
    name : string
        mdf file name
    load_measured_data : bool
        load data option; default *True*

        * if *True* the data group binary data block will be loaded in RAM
        * if *False* the channel data is read from disk on request
    version : string
        mdf file version ('3.00', '3.10', '3.20', '3.30', '4.00', '4.10', '4.11'); default '3.20'

    """
    def __init__(self, name=None, load_measured_data=True, version='3.20'):
        if name and os.path.isfile(name):
            with open(name, 'rb') as file_stream:
                file_stream.read(8)
                version = file_stream.read(4).decode('ascii')
            if version in MDF3_VERSIONS:
                self.file = MDF3(name, load_measured_data)
            elif version in MDF4_VERSIONS:
                self.file = MDF4(name, load_measured_data)
        else:
            if version in MDF3_VERSIONS:
                self.file = MDF3(name, version=version)
            elif version in MDF4_VERSIONS:
                self.file = MDF4(name, version=version)

    def __setattr__(self, attr, value):
        if attr == 'file':
            super(MDF, self).__setattr__(attr, value)
        else:
            setattr(self.file, attr, value)

    def __getattr__(self, attr):
        if attr == 'file':
            return super(MDF, self).__getattr__(attr)
        else:
            return getattr(self.file, attr)

    def convert(self, to):
        """convert MDF to other versions

        Parameters
        ----------
        to : str
            new mdf version from ('3.00', '3.10', '3.20', '3.30', '4.00', '4.10', '4.11')

        Returns
        -------
        out : MDF
            new MDF object

        """
        if not to in MDF3_VERSIONS + MDF4_VERSIONS:
            print('Unknown output mdf version "{}". Available versions are {}'.format(to, MDF4_VERSIONS + MDF3_VERSIONS))
            return
        else:
            out = MDF(version=to)
            if self.version in MDF3_VERSIONS:
                master_type = (V3_MASTER,)
            else:
                master_type = (V4_MASTER, V4_VIRTUAL_MASTER)
            for i, gp in enumerate(self.groups):
                sigs = []
                for j, ch in enumerate(gp['channels']):
                    if not ch['channel_type'] in master_type:
                        sigs.append(self.get(group=i, index=j))
                out.append(sigs,
                           'Converted from {} to {}'.format(self.version, to),
                           common_timebase=True)
            return out

    def export(self, format, filename=None):
        """ export MDF to other formats. The *MDF* file name is used is available,
        else the *filename* aragument must be provided.

        Parameters
        ----------
        format : string
            can be one of the following:

                * *csv* : CSV export that uses the ";" delimiter. This option wil generate a new csv file for each data group (<MDFNAME>_DataGroup_XX.csv).
                * *hdf5* : HDF5 file output; each *MDF* data group is mapped to a *HDF5* group with the name 'Data Group xx' (where xx is the index)
                * *excel* : Excel file output (very slow). This option wil generate a new excel file for each data group (<MDFNAME>_DataGroup_XX.xlsx).

        filename : string
            export file name

        """
        if filename is None and self.name is None:
            warnings.warn('Must specify filename for export if MDF was created without a file name')
        else:
            name = self.name if self.name else filename
            if format == 'hdf5':
                try:
                    from h5py import File as HDF5
                except ImportError:
                    print('h5py not found; export to HDF5 is unavailable')
                    return
                else:
                    name = os.path.splitext(name)[0] + '.hdf'
                    with HDF5(name, 'w') as f:
                        # header information
                        group = f.create_group(os.path.basename(name))

                        if self.version in MDF3_VERSIONS:
                            for item in ('date', 'time', 'author', 'organization', 'project', 'subject'):
                                group.attrs[item] = self.header[item]

                        # save each data group in a HDF5 group called "Data Group xx"
                        # with the index starting from 1
                        # each HDF5 group will have a string attribute "master" that
                        # will hold the name of the master channel
                        for i, grp in enumerate(self.groups):
                            group = f.create_group(r'/' + 'Data Group {}'.format(i + 1))

                            master_index = self.masters_db[i]

                            for j, ch in enumerate(grp['channels']):
                                name = ch.name
                                sig = self.get(group=i, index=j)
                                if j == master_index:
                                    group.attrs['master'] = name
                                dataset = group.create_dataset(name, data=sig.samples)
                                dataset.attrs['unit'] = sig.unit
                                dataset.attrs['comment'] = sig.comment if sig.comment else ''

            elif format == 'excel':
                try:
                    import xlsxwriter
                except ImportError:
                    print('xlsxwriter not found; export to Excel is unavailable')
                    return
                else:
                    excel_name = os.path.splitext(name)[0]
                    nr = len(self.groups)
                    for i, grp in enumerate(self.groups):
                        print('Exporting group {} of {}'.format(i+1, nr))

                        workbook = xlsxwriter.Workbook('{}_{}.xlsx'.format(excel_name, 'DataGroup_{}'.format(i + 1)))
                        bold = workbook.add_format({'bold': True})

                        ws = workbook.add_worksheet("Information")

                        if self.version in MDF3_VERSIONS:
                            for j, item in enumerate(('date', 'time', 'author', 'organization', 'project', 'subject')):
                                ws.write(j, 0, item.title(), bold)
                                ws.write(j, 1, self.header[item].decode('latin-1'))

                            ws = workbook.add_worksheet('Data Group {}'.format(i + 1))

                            ws.write(0, 0, 'Channel', bold)
                            ws.write(1, 0, 'comment', bold)
                            ws.write(2, 0, 'is master', bold)

                            master_index = self.masters_db[i]

                            for j in range(grp['channel_group']['cycles_nr']):
                                ws.write(j+3, 0, str(j))

                            for j, ch in enumerate(grp['channels']):
                                sig = self.get(group=i, index=j)

                                col = j + 1
                                ws.write(0, col, '{} [{}]'.format(sig.name, sig.unit))
                                ws.write(1, col, sig.comment if sig.comment else '')
                                if j == master_index:
                                    ws.write(2, col, 'x')
                                ws.write_column(3, col, sig.samples.astype(str))

                        workbook.close()

            elif format == 'csv':
                csv_name = os.path.splitext(name)[0]
                nr = len(self.groups)
                for i, grp in enumerate(self.groups):
                    print('Exporting group {} of {}'.format(i+1, nr))
                    with open('{}_{}.csv'.format(csv_name, 'DataGroup_{}'.format(i + 1)), 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile, delimiter=';')

                        ch_nr = len(grp['channels'])
                        channels = [self.get(group=i, index=j) for j in range(ch_nr)]

                        master_index = self.masters_db[i]
                        cycles = grp['channel_group']['cycles_nr']

                        writer.writerow(['Channel',] + ['{} [{}]'.format(ch.name, ch.unit) for ch in channels])
                        writer.writerow(['comment',] + [ch.comment for ch in channels])
                        writer.writerow(['is master',] + ['x' if j == master_index else '' for j in range(ch_nr)])

                        vals = [np.array(range(cycles), dtype=np.uint32), ] + [ch.samples for ch in channels]

                        writer.writerows(zip(*vals))

    def filter(self, channels):
        """ return new *MDF* object that contains only the channels listed in *channels* argument

        Parameters
        ----------
        channels : list
            list of channel names to be filtered

        Returns
        -------
        mdf : MDF
            new MDF file

        """

        # group channels by group index
        gps = {}
        for ch in channels:
            if ch in self.channels_db:
                group, index = self.channels_db[ch]
                if not group in gps:
                    gps[group] = []
                gps[group].append(index)
            else:
                message = 'MDF filter error: Channel "{}" not found'.format(ch)
                warnings.warn(message)
                continue

        mdf = MDF(version=self.version)

        # append filtered channels to new MDF
        for group in gps:
            sigs = []
            for index in gps[group]:
                sigs.append(self.get(group=group, index=index))
            mdf.append(sigs,
                       'Signals filtered from <{}>'.format(os.path.basename(self.name)),
                       common_timebase=True)

        return mdf

    def iter_to_pandas(self):
        """ generator that yields channel groups as pandas DataFrames"""
        try:
            from pandas import DataFrame
        except ImportError:
            print('pandas not found; export to pandas DataFrame is unavailable')
            return
        else:
            for i, gp in enumerate(self.groups):
                master_index = self.masters_db[i]
                pandas_dict = {gp['channels'][master_index].name: self.get(group=i, index=master_index, samples_only=True)}
                for j, ch in enumerate(gp['channels']):
                    if j == master_index:
                        continue
                    name = gp['channels'][j].name
                    pandas_dict[name] = self.get(group=i, index=j, samples_only=True)
                yield DataFrame.from_dict(pandas_dict)


if __name__ == '__main__':
    pass

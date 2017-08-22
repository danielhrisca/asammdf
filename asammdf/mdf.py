"""
MDF file format module

"""
import os
import warnings

from pandas import DataFrame

from .mdf3 import MDF3
from .mdf4 import MDF4
from .signal import Signal
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

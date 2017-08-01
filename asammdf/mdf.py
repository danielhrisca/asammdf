"""
MDF file format module

"""
import os

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

    compression : bool
        compression option for data group binary data block; default *False*
    version : string
        mdf file version ('3.00', '3.10', '3.20', '3.30', '4.00', '4.10', '4.11'); default '3.20'

    """
    def __init__(self, name=None, load_measured_data=True, compression=False, version='3.20'):
        if name and os.path.isfile(name):
            with open(name, 'rb') as file_stream:
                file_stream.read(8)
                version = file_stream.read(4).decode('ascii')
            if version in MDF3_VERSIONS:
                self.file = MDF3(name, load_measured_data, compression=compression)
            elif version in MDF4_VERSIONS:
                self.file = MDF4(name, load_measured_data, compression=compression)
        else:
            if version in MDF3_VERSIONS:
                self.file = MDF3(name, compression=compression, version=version)
            elif version in MDF4_VERSIONS:
                self.file = MDF4(name, compression=compression, version=version)

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

    def convert(self, to, compression=False):
        """convert MDF to other versions

        Parameters
        ----------
        to : str
            new mdf version from ('3.00', '3.10', '3.20', '3.30', '4.00', '4.10', '4.11')
        compression : bool
            enable raw channel data compression for out MDF; default *False*

        Returns
        -------
        out : MDF
            new MDF object

        """
        if not to in MDF3_VERSIONS + MDF4_VERSIONS:
            print('Unknown output mdf version "{}". Available versions are {}'.format(to, MDF4_VERSIONS + MDF3_VERSIONS))
            return
        else:
            out = MDF(version=to, compression=compression)
            if self.version in MDF3_VERSIONS:
                master_type = (V3_MASTER,)
            else:
                master_type = (V4_MASTER, V4_VIRTUAL_MASTER)
            for i, gp in enumerate(self.groups):
                sigs = []
                t = self.get_master_data(group=i)
                for j, ch in enumerate(gp['channels']):
                    if not ch['channel_type'] in master_type:
                        vals, name, conversion, unit = self.get_channel_data(group=i, index=j, return_info=True)
                        sigs.append(Signal(samples=vals,
                                           timestamps=t,
                                           unit=unit,
                                           name=name,
                                           conversion=conversion))
                out.append(sigs, 'Converted from {} to {}'.format(self.version, to))
            return out


if __name__ == '__main__':
    pass

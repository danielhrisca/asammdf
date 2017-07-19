"""
MDF file format module

"""

from .mdf3 import MDF3
from .mdf4 import MDF4
import os

class MDF(object):
    """

    Parameters
    ----------

    Attributes
    ----------

    """
    """
    Class that implements MDF version 4 file format

    :param args[0]: file name; optional
    :type args[0]: string
    """
    def __init__(self, name, load_measured_data=True, version='3.20'):
        """"Summary line.

        Extended description of function.

        Parameters
        ----------
        name : Type of file_name
            Description of file_name default None
        empty : Type of empty
            Description of empty default False
        load_measured_data : Type of load_measured_data
            Description of load_measured_data default True
        version : Type of version
            Description of version default '3.20'

        Returns
        -------

        Examples
        --------
        >>>>

        """
        super().__init__()
        if os.path.isfile(name):
            with open(name, 'rb') as file_stream:
                file_stream.read(8)
                version = file_stream.read(4)
            if version in (b'3.00', b'3.10', b'3.20', b'3.30'):
                self.file = MDF3(name, load_measured_data)
            elif version in (b'4.00', b'4.10', b'4.11'):
                self.file = MDF4(name, load_measured_data)
        else:
            if version in ('3.00', '3.10', '3.20', '3.30'):
                self.file = MDF3(name)
            elif version in ('4.00', '4.10', '4.11'):
                self.file = MDF4(name)

    def __setattr__(self, attr, value):
        if attr == 'file':
            super().__setattr__(attr, value)
        else:
            setattr(self.file, attr, value)

    def __getattr__(self, attr):
        if attr == 'file':
            return super().__getattr__(attr)
        else:
            return getattr(self.file, attr)

    @property
    def name(self):
        return self.file.name

    @name.setter
    def name(self, value):
        self.file.name = value


if __name__ == '__main__':
    pass

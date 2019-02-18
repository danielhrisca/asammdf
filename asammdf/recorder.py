# -*- coding: utf-8 -*-
""" MF4 Recorder """

import logging
from pathlib import Path
from datetime import datetime
from threading import Lock, Thread
from time import sleep

from .mdf import MDF
from .blocks.conversion_utils import from_dict


logger = logging.getLogger("asammdf")
WRITE_SIZE = 512 * 1024


__all__ = ["SignalDescription", "Recorder"]


class SignalDescription:

    __slots__ = (
        'name',
        'comment',
        'unit',
        'dtype',
        'conversion',
        'bit_mask',
        'master',
    )

    def __init__(
        self,
        name,
        dtype,
        unit='',
        comment='',
        conversion=None,
        bit_mask=None,
        master=False,
    ):
        self.name = name
        self.dtype = dtype
        self.unit = unit
        self.comment = comment
        self.conversion = from_dict(conversion)
        self.bit_mask = bit_mask
        self.master = master

        self.recording_thread = None


class Recorder(object):
    """ MF4 recorder

    Attributes
    ----------
    frozen : bool
        the recorder is frozen and cannot register new channels from the moment the
        recording is started until the measurement is saved
    mdf : asammdf.MDF
        measurement object
    lock : threading.Lock
        allows safe update of samples and channels selection
    buffers : list
        list of buffer size for each registered channel group

    """

    def __init__(self, plugin):
        self.frozen = False
        self.mdf = MDF()
        self.lock = Lock()
        self.buffers = []
        self.plugin = plugin

    def register_channels(self, channels, source="", comment=""):
        """ registers the channels and creates a new channel group in the underlying
        measurement object

        Parameters
        ----------
        channels : list
            list of SignalDescription objects
        source : str
            source name
        comment : str
            source comment

        Returns
        -------
        index : int
            index of newly created channel group

        """
        if not self.frozen:
            index = len(self.mdf.groups)
            self.mdf._register_channels(channels, source, comment)

            group = self.mdf.groups[-1]

            size = group.types.itemsize
            block_size = WRITE_SIZE // size * size
            self.record_size.append(block_size)
            stream = self.mdf._tempfile

            stream =  self.mdf._tempfile
            address = stream.tell()
            stream.write(b'\0' * block_size)

            group.data_block.append(address)

            group.data_block_addr.append(address)
            group.data_size.append(0)
            group.data_block_size.append(0)

            return index

    def start(self, timestamp=None):
        """ start recording and freezes measurement (no new channels can be registered
        until the measurement is saved)

        Parameters
        ----------
        timestamp : datetime
            start of measurement time stamp; default *None*

        """
        if timestamp is None:
            timestamp = datetime.now()
        self.frozen = True
        self.mdf.header.start_time = timestamp
        self.plugin.start()
        self.thread = Thread(target=self._acquire, args=())
        self.thread.start()
        self.recording = True

    def stop(self):
        """ stop recording

        Parameters
        ----------
        timestamp : datetime
            start of measurement time stamp; default *None*

        """
        self.recording = False
        while self.thread.is_alive():
            sleep(0.005)

    def _acquire(self):
        """ update channel group samples

        Parameters
        ----------
        index : int
            channel group index
        data : bytes
            new raw samples bytes

        """
        seek = self.mdf._tempfile.seek
        write = self.mdf._tempfile.write
        tell = self.mdf._tempfile.tell

        while 1:

            if not self.recording:
                self.plugin.stop()
                break

            self.lock.acquire()

            if not self.plugin.queue.empty():
                index, data = self.plugin.queue.get()
                group = self.mdf.groups[index]

                buffer_limit = self.record_size[index]

                size = len(data)
                cycles = size // group.channel_group.samples_bytes_nr

                while size:
                    address = group.data_block_addr[-1]
                    current_size = group.data_size[-1]

                    if buffer_limit - current_size <= size:
                        # fill the current block
                        seek(address + current_size)

                        write(data[:buffer_limit - current_size])
                        data = data[buffer_limit - current_size:]
                        group.data_size[-1] = buffer_limit
                        group.data_block_size[-1] = buffer_limit

                        # allocate a new block
                        seek(0, 2)
                        address = tell()
                        write(b'\0' * buffer_limit)

                        group.data_block_addr.append(address)
                        group.data_block.append(address)
                        group.data_size.append(0)
                        group.data_block_size.append(0)

                        size -= buffer_limit - current_size

                    else:
                        seek(address + current_size)

                        write(data)
                        group.data_size[-1] += size
                        group.data_block_size[-1] += size

                        size = 0

                group.channel_group.cycles_nr += cycles

                self.lock.release()
            else:
                sleep(0.005)

        while not self.plugin.queue.empty():
            index, data = self.plugin.queue.get()
            group = self.mdf.groups[index]

            buffer_limit = self.record_size[index]

            size = len(data)
            cycles = size // group.channel_group.samples_bytes_nr

            while size:
                address = group.data_block_addr[-1]
                current_size = group.data_size[-1]

                if buffer_limit - current_size <= size:
                    # fill the current block
                    seek(address + current_size)

                    write(data[:buffer_limit - current_size])
                    data = data[buffer_limit - current_size:]
                    group.data_size[-1] = buffer_limit
                    group.data_block_size[-1] = buffer_limit

                    # allocate a new block
                    seek(0, 2)
                    address = tell()
                    write(b'\0' * buffer_limit)

                    group.data_block_addr.append(address)
                    group.data_block.append(address)
                    group.data_size.append(0)
                    group.data_block_size.append(0)

                    size -= buffer_limit - current_size

                else:
                    seek(address + current_size)

                    write(data)
                    group.data_size[-1] += size
                    group.data_block_size[-1] += size

                    size = 0

            group.channel_group.cycles_nr += cycles

            self.lock.release()

    def select(self, channels, record_offset=0):
        """ select Signals from the underlying measurement object

        Parameters
        ----------
        channels : list
            channel selection list (see asammdf.MDF.select documentation)
        record_offset : int
            select channels starting with the given record offset

        Returns
        -------
        channels : list
            list of *asammdf.Signal* objects (see asammdf.MDF.select documentation)
        """

        self.lock.acquire()
        self.mdf._master_channel_cache.clear()
        channels = self.mdf.select(channels, record_offset=record_offset)
        self.lock.release()
        return channels

    def save(self, dst, overwrite=False, compression=0):
        """ save underlying measurement. Once saved, the measurement object is unfrozen
        and reset for a new recording

        Paremeters
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

        """
        name = Path(dst)
        self.mdf.save(name, overwrite=overwrite, compression=compression)
        for group in self.mdf.groups:
            group.data_block = []

            group.channel_group.cycles_nr = 0
            group.data_block_addr = []
            group.data_size = []
            group.data_block_size = []
        self.frozen = False


if __name__ == "__main__":
    pass

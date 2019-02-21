# -*- coding: utf-8 -*-
""" MF4 Recorder """

import logging
from datetime import datetime
from threading import Thread
from queue import Queue
from time import sleep

from abc import ABC, abstractmethod


logger = logging.getLogger("asammdf")

__all__ = ["DeviceBase", ]


class DeviceBase(ABC):

    def __init__(self, plugin):
        self.queue = Queue()
        self.thread = None
        self.running = False

    @abstractmethod
    def start(self):
        """ setup acquisition and start receiving data by starting
        the acquisition thread

        """
        self.thread = Thread(target=self._acquire, args=())
        self.thread.start()
        self.running = True

    @abstractmethod
    def stop(self):
        """ stop receiving data and stop acqusition thread

        """
        self.running = False
        while self.thread.is_alive():
            sleep(0.005)

    @abstractmethod
    def _acquire(self):
        """ setup acquisition and start receiving data by starting
        the acquisition thread

        """
        while True:

            if self.running:
                # get data from acquisition device and place it in the queue
                self.queue.put((group_index, data_bytes))
            else:
                break


if __name__ == "__main__":
    pass

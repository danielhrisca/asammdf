# -*- coding: utf-8 -*-
""" ASAM MDF version 2 file format module """

from __future__ import division, print_function

from .mdf_v3 import MDF3
from .utils import (
    validate_memory_argument,
    validate_version_argument,
)


__all__ = ['MDF2', ]


# MDF versions 2 and 3 share the same implementation
class MDF2(MDF3):

    _terminate = False

    """ shared implementation for mdf version 2 and 3 """

    def __init__(self, name=None, memory='full', version='2.14', callback=None):
        memory = validate_memory_argument(memory)
        version = validate_version_argument(version, hint=2)

        super(MDF2, self).__init__(name, memory, version, callback)


if __name__ == '__main__':
    pass

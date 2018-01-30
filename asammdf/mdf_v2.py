# -*- coding: utf-8 -*-
""" ASAM MDF version 2 file format module """

from __future__ import division, print_function

from .mdf_v3 import MDF3


__all__ = ['MDF2', ]


# MDF versions 2 and 3 share the same implementation
class MDF2(MDF3):
    pass


if __name__ == '__main__':
    pass

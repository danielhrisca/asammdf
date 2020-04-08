# -*- coding: utf-8 -*-
"""
asammdf utility functions for source information
"""

from . import v2_v3_blocks as v3b
from . import v2_v3_constants as v3c
from . import v4_blocks as v4b
from . import v4_constants as v4c

__all__ = [
    "Source",
]


class Source:

    __slots__ = "name", "path", "comment", "source_type", "bus_type"

    def __init__(self, name, path, comment, source_type, bus_type):
        """ Commons reprezentation for source information

        Attributes
        ----------
        name : str
            source name
        path : str
            source path
        comment : str
            source comment
        source_type : int
            source type code
        bus_type : int
            source bus code

        """
        self.name, self.path, self.comment, self.source_type, self.bus_type = (
            name,
            path,
            comment,
            source_type,
            bus_type,
        )

    @classmethod
    def from_source(cls, source):
        if isinstance(source, v3b.ChannelExtension):
            if source.type == v3c.SOURCE_ECU:
                source = cls(
                    source.name,
                    source.path,
                    source.comment,
                    0,  # source type other
                    0,  # bus type none
                )
            else:
                source = cls(
                    source.name,
                    source.path,
                    source.comment,
                    2,  # source type bus
                    2,  # bus type CAN
                )

        elif isinstance(source, v4b.SourceInformation):
            return cls(
                source.name,
                source.path,
                source.comment,
                source.source_type,
                source.bus_type,
            )
        elif isinstance(source, Source):
            return cls(
                source.name,
                source.path,
                source.comment,
                source.source_type,
                source.bus_type,
            )

# -*- coding: utf-8 -*-
""" ASAM MDF version 2 file format module """

from __future__ import annotations

from io import BufferedReader, BytesIO

from ..types import StrPathType
from .mdf_v3 import MDF3
from .utils import validate_version_argument

__all__ = ["MDF2"]


# MDF versions 2 and 3 share the same implementation
class MDF2(MDF3):

    _terminate = False

    """ shared implementation for mdf version 2 and 3 """

    def __init__(
        self,
        name: BufferedReader | BytesIO | StrPathType | None = None,
        version: str = "2.14",
        **kwargs
    ) -> None:
        version = validate_version_argument(version, hint=2)

        super().__init__(name, version, **kwargs)


if __name__ == "__main__":
    pass

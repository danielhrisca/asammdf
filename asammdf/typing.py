from bz2 import BZ2File
from gzip import GzipFile
from io import BytesIO
from os import PathLike
from typing import Any, Optional, Sequence, Tuple, Union
from zipfile import ZipFile

from canmatrix import CanMatrix
from numpy.typing import NDArray
from typing_extensions import Literal

from .blocks import v2_v3_blocks, v4_blocks

StrOrBytesPath = Union[str, bytes, PathLike[str], PathLike[bytes]]

ChannelGroupType = Union[v2_v3_blocks.ChannelGroup, v4_blocks.ChannelGroup]
ChannelsType = Union[
    Sequence[str], Sequence[Tuple[Optional[str], int, int]], Sequence[Tuple[str, int]]
]
DbcFileType = Tuple[Union[StrOrBytesPath, CanMatrix], int]
EmptyChannelsType = Literal["skip", "zeros"]
InputType = Union[BytesIO, StrOrBytesPath, ZipFile, BZ2File, GzipFile]
RasterType = Union[float, str, NDArray[Any]]

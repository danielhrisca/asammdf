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
from .blocks.source_utils import Source

StrPath = Union[str, "PathLike[str]"]

# asammdf specific types

ChannelConversionType = Union[
    v2_v3_blocks.ChannelConversion, v4_blocks.ChannelConversion
]
ChannelGroupType = Union[v2_v3_blocks.ChannelGroup, v4_blocks.ChannelGroup]
ChannelsType = Union[
    Sequence[str], Sequence[Tuple[Optional[str], int, int]], Sequence[Tuple[str, int]]
]
DbcFileType = Tuple[Union[StrPath, CanMatrix], int]
EmptyChannelsType = Literal["skip", "zeros"]
FloatInterpolationModeType = Literal[0, 1]
InputType = Union[BytesIO, StrPath, ZipFile, BZ2File, GzipFile]
IntInterpolationModeType = Literal[0, 1, 2]
RasterType = Union[float, str, NDArray[Any]]
SourceType = Union[v2_v3_blocks.ChannelExtension, v4_blocks.SourceInformation, Source]
SyncType = Literal[0, 1, 2, 3, 4]

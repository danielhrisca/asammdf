from bz2 import BZ2File
from gzip import GzipFile
from io import BytesIO
from os import PathLike
from typing import Any, Optional, Sequence, Tuple, TYPE_CHECKING, Union
from zipfile import ZipFile

from canmatrix import CanMatrix
from numpy.typing import NDArray
from typing_extensions import Literal

from .blocks import v2_v3_blocks, v4_blocks
from .blocks.source_utils import Source

if TYPE_CHECKING:
    from .blocks.mdf_v2 import MDF2
    from .blocks.mdf_v3 import MDF3
    from .blocks.mdf_v4 import MDF4

StrPath = Union[str, "PathLike[str]"]
StrOrBytesPath = Union[str, bytes, "PathLike[str]", "PathLike[bytes]"]

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
MDF_v2_v3_v4 = Union["MDF2", "MDF3", "MDF4"]
RasterType = Union[float, str, NDArray[Any]]
SourceType = Union[v2_v3_blocks.ChannelExtension, v4_blocks.SourceInformation, Source]
SyncType = Literal[0, 1, 2, 3, 4]

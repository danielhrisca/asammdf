from bz2 import BZ2File
from collections.abc import Sequence
from gzip import GzipFile
from io import BufferedRandom, BufferedReader, BufferedWriter, BytesIO
from mmap import mmap
from os import PathLike
from typing import Any, Literal, TYPE_CHECKING, Union
from zipfile import ZipFile

from canmatrix import CanMatrix
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .blocks import v2_v3_blocks, v4_blocks, v4_constants
    from .blocks.mdf_v2 import MDF2
    from .blocks.mdf_v3 import MDF3
    from .blocks.mdf_v4 import MDF4
    from .blocks.source_utils import Source


StrPathType = Union[str, "PathLike[str]"]
StrOrBytesPathType = Union[str, bytes, "PathLike[str]", "PathLike[bytes]"]
ReadableBufferType = BufferedRandom | BufferedReader | BytesIO | mmap
WritableBufferType = BufferedRandom | BufferedWriter | BytesIO | mmap

# asammdf specific types

BusType = Literal["CAN", "LIN"]
ChannelConversionType = Union["v2_v3_blocks.ChannelConversion", "v4_blocks.ChannelConversion"]
ChannelGroupType = Union["v2_v3_blocks.ChannelGroup", "v4_blocks.ChannelGroup"]
ChannelsType = Sequence[str] | Sequence[tuple[str | None, int, int]] | Sequence[tuple[str, int]]
ChannelType = Union["v2_v3_blocks.Channel", "v4_blocks.Channel"]
CompressionType = Union[Literal[0, 1, 2, 3, 4, 5], "v4_constants.CompressionAlgorithm"]
DataGroupType = Union["v2_v3_blocks.DataGroup", "v4_blocks.DataGroup"]
DbcFileType = tuple[StrPathType | CanMatrix, int]
EmptyChannelsType = Literal["skip", "zeros"]
FloatInterpolationModeType = Literal[0, 1]
InputType = BufferedReader | BytesIO | StrPathType | ZipFile | BZ2File | GzipFile
IntInterpolationModeType = Literal[0, 1, 2]
MDF_v2_v3_v4 = Union["MDF2", "MDF3", "MDF4"]
RasterType = float | str | NDArray[Any]
SourceType = Union["v2_v3_blocks.ChannelExtension", "v4_blocks.SourceInformation", "Source"]
SyncType = Literal[0, 1, 2, 3, 4]

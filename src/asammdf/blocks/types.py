from collections.abc import Sequence
from os import PathLike
from typing import Literal, TYPE_CHECKING, Union

from canmatrix import CanMatrix
from numpy.typing import NDArray
from typing_extensions import Any

if TYPE_CHECKING:
    from . import v2_v3_blocks as v3b
    from . import v4_blocks as v4b
    from . import v4_constants as v4c
    from .source_utils import Source


StrPath = str | PathLike[str]

# asammdf specific types

BusType = Literal["CAN", "LIN"]
ChannelConversionType = Union["v3b.ChannelConversion", "v4b.ChannelConversion"]
ChannelsType = Sequence[str | tuple[str | None, int, int] | tuple[str, int]]
CompressionType = Union[Literal[0, 1, 2, 3, 4, 5], "v4c.CompressionAlgorithm"]
DbcFileType = tuple[StrPath | CanMatrix, int]
EmptyChannelsType = Literal["skip", "zeros"]
FloatInterpolationModeType = Literal[0, 1]
IntInterpolationModeType = Literal[0, 1, 2]
RasterType = float | str | NDArray[Any]
SourceType = Union["v3b.ChannelExtension", "v4b.SourceInformation", "Source"]

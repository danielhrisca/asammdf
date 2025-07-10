"""ASAM MDF version 2 file format module"""

from typing_extensions import Unpack

from .mdf_v3 import Kwargs, MDF3
from .types import StrPath
from .utils import FileLike, MdfException, validate_version_argument
from .v2_v3_constants import Version2

__all__ = ["MDF2"]


# MDF versions 2 and 3 share the same implementation
class MDF2(MDF3):
    """Shared implementation for MDF version 2 and 3."""

    def __init__(
        self,
        name: StrPath | FileLike | None = None,
        version: Version2 = "2.14",
        channels: list[str] | None = None,
        **kwargs: Unpack[Kwargs],
    ) -> None:
        version = validate_version_argument(version, hint=2)

        if not kwargs.get("__internal__", False):
            raise MdfException("Always use the MDF class; do not use the class MDF2 directly")

        super().__init__(name, version, channels=channels, **kwargs)


if __name__ == "__main__":
    pass

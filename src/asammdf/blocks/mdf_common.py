from abc import ABC
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
import logging
from pathlib import Path
from typing import Generic

import numpy as np
from numpy.typing import DTypeLike, NDArray
from typing_extensions import Any, Required, TypedDict, TypeVar

from . import v2_v3_blocks as v3b
from . import v4_blocks as v4b
from .types import DbcFileType, StrPath
from .utils import (
    ChannelsDB,
    DataBlockInfo,
    EMPTY_TUPLE,
    MdfException,
    SignalDataBlockInfo,
)

logger = logging.getLogger("asammdf")

__all__ = ["MDF_Common"]


class MdfKwargs(TypedDict, total=False):
    temporary_folder: StrPath | None
    raise_on_multiple_occurrences: bool
    use_display_names: bool
    fill_0_for_missing_computation_channels: bool
    remove_source_from_channel_names: bool
    password: str | None
    progress: Callable[[int, int], None] | Any
    callback: Callable[[int, int], None] | Any


class MdfCommonKwargs(MdfKwargs, total=False):
    original_name: Required[str | Path | None]
    __internal__: bool


_DG = TypeVar("_DG", v3b.DataGroup, v4b.DataGroup)
_CG = TypeVar("_CG", v3b.ChannelGroup, v4b.ChannelGroup)
_CN = TypeVar("_CN", v3b.Channel, v4b.Channel)
_CD = TypeVar("_CD", v3b.ChannelDependency, list[v4b.ChannelArrayBlock] | list[tuple[int, int]])
_ST = TypeVar("_ST", list[int], list[int] | list[tuple[int, int]] | np.dtype[Any])


class Group(Generic[_DG, _CG, _CN, _CD, _ST]):
    __slots__ = (
        "channel_dependencies",
        "channel_group",
        "channels",
        "data_blocks",
        "data_blocks_info_generator",
        "data_group",
        "data_location",
        "index",
        "read_split_count",
        "record",
        "record_size",
        "signal_data",
        "signal_types",
        "single_channel_dtype",
        "sorted",
        "string_dtypes",
        "trigger",
        "uses_ld",
        "uuid",
    )

    def __init__(self, data_group: _DG) -> None:
        self.data_group: _DG = data_group
        self.channel_group: _CG
        self.channels: list[_CN] = []
        self.channel_dependencies: list[_CD | None] = []
        self.signal_data: list[tuple[list[SignalDataBlockInfo], Iterator[SignalDataBlockInfo]] | None] = []
        self.record: list[tuple[np.dtype[Any], int, int, int] | None] | None = None
        self.record_size: dict[int, int] = {}
        self.trigger: v3b.TriggerBlock | None = None
        self.sorted: bool
        self.string_dtypes: list[np.dtype[np.bytes_]] = []
        self.data_blocks: list[DataBlockInfo] = []
        self.signal_types: _ST
        self.single_channel_dtype: DTypeLike | None = None
        self.uses_ld = False
        self.read_split_count = 0
        self.data_blocks_info_generator: Iterator[DataBlockInfo] = iter(EMPTY_TUPLE)
        self.uuid = ""
        self.data_location: int
        self.index = 0

    def set_blocks_info(self, info: list[DataBlockInfo]) -> None:
        self.data_blocks = info

    def __contains__(self, item: str) -> bool:
        return hasattr(self, item)

    def clear(self) -> None:
        self.data_blocks.clear()
        self.channels.clear()
        self.channel_dependencies.clear()
        self.signal_data.clear()
        self.data_blocks_info_generator = iter(())

    def get_data_blocks(self) -> Iterator[DataBlockInfo]:
        yield from self.data_blocks

        while True:
            try:
                info = next(self.data_blocks_info_generator)
                self.data_blocks.append(info)
                yield info
            except StopIteration:
                break

    def get_signal_data_blocks(self, index: int) -> Iterator[SignalDataBlockInfo]:
        signal_data = self.signal_data[index]
        if signal_data is not None:
            signal_data_blocks, signal_generator = signal_data
            yield from signal_data_blocks

            while True:
                try:
                    info = next(signal_generator)
                    signal_data_blocks.append(info)
                    yield info
                except StopIteration:
                    break

    def load_all_data_blocks(self) -> None:
        for _ in self.get_data_blocks():
            continue


GroupV3 = Group[v3b.DataGroup, v3b.ChannelGroup, v3b.Channel, v3b.ChannelDependency, list[int]]
GroupV4 = Group[
    v4b.DataGroup,
    v4b.ChannelGroup,
    v4b.Channel,
    list[v4b.ChannelArrayBlock] | list[tuple[int, int]],
    list[int] | list[tuple[int, int]] | np.dtype[Any],
]

_Group = TypeVar("_Group", GroupV3, GroupV4)


class MDF_Common(ABC, Generic[_Group]):
    """Common methods for MDF objects."""

    def __init__(self, raise_on_multiple_occurrences: bool) -> None:
        self.groups: list[_Group] = []
        self.channels_db = ChannelsDB()
        self._raise_on_multiple_occurrences = raise_on_multiple_occurrences

    def _set_temporary_master(self, master: NDArray[Any] | None) -> None:
        self._master = master

    # @lru_cache(maxsize=1024)
    def _validate_channel_selection(
        self,
        name: str | None = None,
        group: int | None = None,
        index: int | None = None,
    ) -> tuple[int, int]:
        """Validate channel selection.

        The channel can be specified in two ways:

        * Using the first positional argument `name`.

          * If there are multiple occurrences for this channel, then the `group`
            and `index` arguments can be used to select a specific group.
          * If there are multiple occurrences for this channel and either the
            `group` or `index` arguments is None, then a warning is issued.

        * Using the group number (keyword argument `group`) and the channel
          number (keyword argument `index`). Use `info` method for group and
          channel numbers.

        Parameters
        ----------
        name : str, optional
            Name of channel.
        group : int, optional
            0-based group index.
        index : int, optional
            0-based channel index.

        Returns
        -------
        group_index, channel_index : (int, int)
            Selected channel's group and channel index.

        Raises
        ------
        MdfException
            * if the channel name is not found
            * if the group index is out of range
            * if the channel index is out of range
            * if there are multiple channel occurrences in the file and the
              arguments `name`, `group`, `index` are ambiguous. This behaviour
              can be turned off by setting `raise_on_multiple_occurrences` to
              False.
        """

        if name is None:
            if group is None or index is None:
                message = 'Invalid arguments for channel selection: must give "name" or, "group" and "index"'
                raise MdfException(message)
            else:
                gp_nr, ch_nr = group, index
                if ch_nr >= 0:
                    try:
                        grp = self.groups[gp_nr]
                    except IndexError:
                        raise MdfException("Group index out of range") from None

                    try:
                        grp.channels[ch_nr]
                    except IndexError:
                        raise MdfException(f"Channel index out of range: {(name, group, index)}") from None
        else:
            if name not in self.channels_db:
                raise MdfException(f'Channel "{name}" not found')
            else:
                if group is None:
                    entries = self.channels_db[name]
                    if len(entries) > 1:
                        if self._raise_on_multiple_occurrences:
                            message = (
                                f'Multiple occurrences for channel "{name}": {entries}. '
                                'Provide both "group" and "index" arguments'
                                " to select another data group"
                            )
                            logger.exception(message)
                            raise MdfException(message)
                        else:
                            message = (
                                f'Multiple occurrences for channel "{name}": {entries}. '
                                "Returning the first occurrence since the MDF object was "
                                "configured to not raise an exception in this case."
                            )
                            logger.warning(message)
                            gp_nr, ch_nr = entries[0]
                    else:
                        gp_nr, ch_nr = entries[0]

                else:
                    if index is not None and index < 0:
                        gp_nr = group
                        ch_nr = index
                    else:
                        if index is None:
                            entries = tuple((gp_nr, ch_nr) for gp_nr, ch_nr in self.channels_db[name] if gp_nr == group)
                            count = len(entries)

                            if count == 1:
                                gp_nr, ch_nr = entries[0]

                            elif count == 0:
                                message = f'Channel "{name}" not found in group {group}'
                                raise MdfException(message)

                            else:
                                if self._raise_on_multiple_occurrences:
                                    message = (
                                        f'Multiple occurrences for channel "{name}": {entries}. '
                                        'Provide both "group" and "index" arguments'
                                        " to select another data group"
                                    )
                                    logger.exception(message)
                                    raise MdfException(message)
                                else:
                                    message = (
                                        f'Multiple occurrences for channel "{name}": {entries}. '
                                        "Returning the first occurrence since the MDF object was "
                                        "configured to not raise an exception in this case."
                                    )
                                    logger.warning(message)
                                    gp_nr, ch_nr = entries[0]
                        else:
                            if (group, index) in self.channels_db[name]:
                                ch_nr = index
                                gp_nr = group
                            else:
                                message = f'Channel "{name}" not found in group {group} at index {index}'
                                raise MdfException(message)

        return gp_nr, ch_nr


class _BusInfo(TypedDict):
    dbc_files: Iterable[DbcFileType]
    unknown_id_count: int


class CanInfo(_BusInfo):
    total_unique_ids: set[tuple[int, bool]]
    not_found_ids: defaultdict[StrPath, list[tuple[tuple[int, bool] | int, str]]]
    found_ids: defaultdict[StrPath, set[tuple[tuple[int, int, bool], str]]]
    unknown_ids: set[int | tuple[int, bool]]
    max_flags: list[list[list[bool]]]


class LinInfo(_BusInfo):
    total_unique_ids: set[tuple[int, ...]]
    not_found_ids: defaultdict[StrPath, list[tuple[int, str]]]
    found_ids: defaultdict[StrPath, set[tuple[tuple[int, bool, bool], str]]]
    unknown_ids: set[int]


class LastCallInfo(TypedDict, total=False):
    CAN: CanInfo
    LIN: LinInfo

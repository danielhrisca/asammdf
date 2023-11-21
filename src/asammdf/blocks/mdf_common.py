"""
ASAM MDF version 4 file format module
"""

from __future__ import annotations

import logging
from typing import Any

from numpy.typing import NDArray

from .utils import MdfException

logger = logging.getLogger("asammdf")

__all__ = ["MDF_Common"]


class MDF_Common:
    """common methods for MDF objects"""

    def _set_temporary_master(self, master: NDArray[Any] | None) -> None:
        self._master = master

    # @lru_cache(maxsize=1024)
    def _validate_channel_selection(
        self,
        name: str | None = None,
        group: int | None = None,
        index: int | None = None,
    ) -> tuple[int, int]:
        """Gets channel comment.
        Channel can be specified in two ways:

        * using the first positional argument *name*

            * if there are multiple occurrences for this channel then the
            *group* and *index* arguments can be used to select a specific
            group.
            * if there are multiple occurrences for this channel and either the
            *group* or *index* arguments is None then a warning is issued

        * using the group number (keyword argument *group*) and the channel
        number (keyword argument *index*). Use *info* method for group and
        channel numbers


        Parameters
        ----------
        name : string
            name of channel
        group : int
            0-based group index
        index : int
            0-based channel index

        Returns
        -------
        group_index, channel_index : (int, int)
            selected channel's group and channel index

        """

        if name is None:
            if group is None or index is None:
                message = "Invalid arguments for channel selection: " 'must give "name" or, "group" and "index"'
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

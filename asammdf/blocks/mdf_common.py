"""
ASAM MDF version 4 file format module
"""

import logging

from .utils import MdfException

logger = logging.getLogger("asammdf")

__all__ = ["MDF_Common"]


class MDF_Common:
    """common methods for MDF objects"""

    def _get_source_names(self, gp_idx, cn_idx):
        group = self.groups[gp_idx]
        cn_source_name = group.channels[cn_idx].source.name
        cg_source_name = (
            group.channel_group.acq_source.name if self.version >= "4.00" else None
        )
        return cn_source_name, cg_source_name

    def _get_source_paths(self, gp_idx, cn_idx):
        group = self.groups[gp_idx]
        cn_source_path = group.channels[cn_idx].source.path
        cg_source_path = (
            group.channel_group.acq_source.path if self.version >= "4.00" else None
        )
        return cn_source_path, cg_source_path

    def _filter_occurences(self, occurences, source_name=None, source_path=None):
        occurences = (
            (gp_idx, cn_idx)
            for gp_idx, cn_idx in occurences
            if (
                source_name is None
                or source_name in self._get_source_names(gp_idx, cn_idx)
            )
            and (
                source_path is None
                or source_path in self._get_source_paths(gp_idx, cn_idx)
            )
        )
        return occurences

    def _set_temporary_master(self, master):
        self._master = master

    # @lru_cache(maxsize=1024)
    def _validate_channel_selection(
        self, name=None, group=None, index=None, source=None
    ):
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
        source : str
            can be used for multiple occurence of the same channel name to
            filter the target channel

        Returns
        -------
        group_index, channel_index : (int, int)
            selected channel's group and channel index

        """

        if name is None:
            if group is None or index is None:
                message = (
                    "Invalid arguments for channel selection: "
                    'must give "name" or, "group" and "index"'
                )
                raise MdfException(message)
            else:
                gp_nr, ch_nr = group, index
                if ch_nr >= 0:
                    try:
                        grp = self.groups[gp_nr]
                    except IndexError:
                        raise MdfException("Group index out of range")

                    try:
                        grp.channels[ch_nr]
                    except IndexError:
                        raise MdfException(
                            f"Channel index out of range: {(name, group, index)}"
                        )
        else:
            if name not in self.channels_db:
                raise MdfException(f'Channel "{name}" not found')
            else:
                if source is not None:
                    for gp_nr, ch_nr in self.channels_db[name]:
                        if source in self._get_source_names(gp_nr, ch_nr):
                            break
                    else:
                        raise MdfException(f"{name} with source {source} not found")
                elif group is None:

                    entries = self.channels_db[name]
                    if len(entries) > 1:
                        message = (
                            f'Multiple occurances for channel "{name}": {entries}. '
                            'Provide both "group" and "index" arguments'
                            " to select another data group"
                        )
                        logger.exception(message)
                        raise MdfException(message)
                    else:
                        gp_nr, ch_nr = entries[0]

                else:
                    if index is not None and index < 0:
                        gp_nr = group
                        ch_nr = index
                    else:
                        if index is None:
                            entries = [
                                (gp_nr, ch_nr)
                                for gp_nr, ch_nr in self.channels_db[name]
                                if gp_nr == group
                            ]
                            count = len(entries)

                            if count == 1:
                                gp_nr, ch_nr = entries[0]

                            elif count == 0:
                                message = f'Channel "{name}" not found in group {group}'
                                raise MdfException(message)

                            else:
                                message = (
                                    f'Multiple occurances for channel "{name}" in group {group}. '
                                    'Provide also the "index" argument'
                                    " to select the desired channel"
                                )
                                raise MdfException(message)
                        else:
                            if (group, index) in self.channels_db[name]:
                                ch_nr = index
                                gp_nr = group
                            else:
                                message = f'Channel "{name}" not found in group {group} at index {index}'
                                raise MdfException(message)

        return gp_nr, ch_nr

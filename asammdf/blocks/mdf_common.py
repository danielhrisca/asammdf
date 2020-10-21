"""
ASAM MDF version 4 file format module
"""

from functools import lru_cache
import logging

from .utils import (
    all_blocks_addresses,
    as_non_byte_sized_signed_int,
    CHANNEL_COUNT,
    ChannelsDB,
    CONVERT,
    count_channel_groups,
    DataBlockInfo,
    debug_channel,
    extract_cncomment_xml,
    fmt_to_datatype_v4,
    get_fmt_v4,
    Group,
    InvalidationBlockInfo,
    is_file_like,
    load_can_database,
    MdfException,
)

logger = logging.getLogger("asammdf")

__all__ = ["MDF_Common"]



class MDF_Common:
    """common methods for MDF objects

    """

    def _get_source_name(self, group, index):
        source = self.groups[group].channels[index].source
        cn_source = source.name if source else ""

        if self.version >= '4.00':
            source = self.groups[group].channel_group.acq_source
            cg_source = source.name if source else ""
            return (cn_source, cg_source)
        else:
            return (cn_source,)

    def _set_temporary_master(self, master):
        self._master = master

    @lru_cache(maxsize=1024)
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
                        if source in self._get_source_name(gp_nr, ch_nr):
                            break
                    else:
                        raise MdfException(f"{name} with source {source} not found")
                elif group is None:

                    gp_nr, ch_nr = self.channels_db[name][0]
                    if len(self.channels_db[name]) > 1:
                        message = (
                            f'Multiple occurances for channel "{name}". '
                            f"Using first occurance from data group {gp_nr}. "
                            'Provide both "group" and "index" arguments'
                            " to select another data group"
                        )
                        logger.warning(message)

                else:
                    if index is not None and index < 0:
                        gp_nr = group
                        ch_nr = index
                    else:
                        for gp_nr, ch_nr in self.channels_db[name]:
                            if gp_nr == group:
                                if index is None:
                                    break
                                elif index == ch_nr:
                                    break
                        else:
                            if index is None:
                                message = f'Channel "{name}" not found in group {group}'
                            else:
                                message = f'Channel "{name}" not found in group {group} at index {index}'
                            raise MdfException(message)

        return gp_nr, ch_nr


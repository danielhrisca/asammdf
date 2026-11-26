import numpy as np

from asammdf import MDF, Signal
from asammdf.gui.widgets.channel_group_info import ChannelGroupInfoWidget
from test.asammdf.gui.test_base import TestBase


class TestChannelGroupInfoWidget(TestBase):
    def test_entries_with_structure_channel(self):
        """
        Test Scope:
            - Ensure that each listed channel points to itself when the group holds a structure channel.
        Events:
            - Append a structure channel and a plain channel to one channel group.
            - Open the channel group info widget for this group.
        Evaluate:
            - Evaluate that the entry of each listed channel selects a channel with the same name.
        """
        timestamps = np.arange(5, dtype="f8")
        structure = np.rec.fromarrays([np.arange(5, dtype="u1"), np.arange(5, dtype="u2")], names=["a", "b"])

        with MDF(version="4.10") as mdf:
            mdf.append(
                [
                    Signal(structure, timestamps, name="Structure"),
                    Signal(np.arange(5, dtype="i4"), timestamps, name="Plain"),
                ]
            )
            group = mdf.groups[0]
            widget = ChannelGroupInfoWidget(mdf, group)

            names = [widget.channels.item(row).name for row in range(widget.channels.count())]
            self.assertIn("Plain", names)

            for row in range(widget.channels.count()):
                item = widget.channels.item(row)
                with self.subTest(item.name):
                    self.assertEqual(item.name, group.channels[item.entry].name)

            widget.close()

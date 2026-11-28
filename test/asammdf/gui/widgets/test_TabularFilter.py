from datetime import datetime, UTC
from unittest import mock
from zoneinfo import ZoneInfo

import pandas as pd
from PySide6 import QtCore

from asammdf.gui.widgets.tabular import Tabular
from test.asammdf.gui.test_base import TestBase


class TestTabularFilter(TestBase):
    def test_filter_timestamps_as_date(self):
        """
        Test Scope:
            - Ensure that filters on the timestamps shown as date keep the rows with the shown values.
        Events:
            - Open a Tabular window with one row in summer time and one row in winter time.
            - Show the time as date in the Europe/Berlin time zone.
            - Filter the timestamps column for the shown value of each row, for the range between them,
              and for a local time that is skipped by the change to summer time.
        Evaluate:
            - Evaluate the shown values.
            - Evaluate that each filter keeps only the expected row.
            - Evaluate that the skipped local time gives a filter error and keeps the previous rows.
        """
        start = datetime(2024, 1, 1, tzinfo=UTC)
        offsets = [(datetime(2024, month, 1, 12, tzinfo=UTC) - start).total_seconds() for month in (7, 12)]
        df = pd.DataFrame({"x": [1.0, 2.0]}, index=pd.Index(offsets, name="timestamps"))

        with mock.patch("asammdf.gui.widgets.tabular.LOCAL_TIMEZONE", ZoneInfo("Europe/Berlin")):
            widget = Tabular(df, start=start)
        widget.time_as_date.setCheckState(QtCore.Qt.CheckState.Checked)

        shown = [str(timestamp) for timestamp in widget.tree.pgdf.df_unfiltered["timestamps"]]
        self.assertListEqual(["2024-07-01 14:00:00+02:00", "2024-12-01 13:00:00+01:00"], shown)

        for conditions, expected in (
            ([("==", "2024-07-01 14:00:00")], [1.0]),
            ([("==", "2024-12-01 13:00:00")], [2.0]),
            ([(">=", "2024-07-01 14:00:00"), ("<", "2024-12-01 13:00:00")], [1.0]),
            ([(">=", "2024-03-31 02:30:00")], None),
        ):
            with self.subTest(conditions):
                # Event
                widget.filters.clear()
                for op, target in conditions:
                    widget.add_filter()
                    filter = widget.filters.itemWidget(widget.filters.item(widget.filters.count() - 1))
                    filter.column.setCurrentText("timestamps")
                    filter.op.setCurrentText(op)
                    filter.target.setText(target)
                    filter.validate_target()
                widget.apply_filters()

                # Evaluate
                if expected is None:
                    # the filter error names the entered time
                    self.assertIn(target, widget.query.toPlainText())
                    self.assertListEqual([1.0], widget.tree.pgdf.df["x"].tolist())
                else:
                    self.assertListEqual(expected, widget.tree.pgdf.df["x"].tolist())

        widget.close()

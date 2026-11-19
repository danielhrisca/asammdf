#!/usr/bin/env python
import pathlib
from unittest import mock

from PySide6 import QtCore, QtTest, QtWidgets

from asammdf import MDF, Signal, Source
from test.asammdf.gui.test_base import OpenMDF
from test.asammdf.gui.widgets.test_BaseFileWidget import TestFileWidget

# Note: If it's possible and make sense, use self.subTests
# to avoid initializing widgets multiple times and consume time.


class TestTabModifyAndExport(TestFileWidget):
    def test_lab_filter_round_trip(self):
        path = pathlib.Path(self.test_workspace, "axle.mf4")
        lab = path.with_suffix(".lab")
        speed = Signal(name="ABS_FrontAxleSpeed", samples=[12], timestamps=[0.0])
        literal = Signal(name="ABS —— None", samples=[1], timestamps=[0.0])
        abs_signal = Signal(name="ABS", samples=[1], timestamps=[0.0])
        abs_can = Signal(name="ABS —— CAN", samples=[2], timestamps=[0.0])
        with MDF() as mdf:
            mdf.append([speed, literal, abs_signal, abs_can])
            mdf.append(
                [speed, abs_signal], acq_source=Source("CAN", "CAN —— None", "", Source.SOURCE_BUS, Source.BUS_TYPE_CAN)
            )
            mdf.save(path)
        self.addCleanup(self.settings.setValue, "test/natural_sort_old", self.settings.value("test/natural_sort_old"))
        self.settings.setValue("test/natural_sort_old", False)
        self.setUpFileWidget(measurement_file=path, default=True)
        self.widget.aspects.setCurrentIndex(1)
        self.widget.filter_view.setCurrentText("Natural sort")
        entry = self.widget.mdf.whereis(speed.name)[0]
        item = next(
            item
            for index in range(self.widget.filter_tree.topLevelItemCount())
            if (item := self.widget.filter_tree.topLevelItem(index)).entry == entry
        )
        item.setCheckState(0, QtCore.Qt.CheckState.Checked)

        with (
            mock.patch.object(QtWidgets.QFileDialog, "getSaveFileName", return_value=(str(lab), "")),
            mock.patch.object(QtWidgets.QInputDialog, "getText", return_value=("lab", True)),
        ):
            self.widget.save_filter_list()

        self.widget.clear_filter()
        self.widget.load_filter_list(file_name=lab)
        with self.subTest("new export"):
            self.assertEqual(lab.read_text().splitlines()[-1], speed.name)
            self.assertEqual(self.widget._selected_filter, {entry: speed.name})

        # "ABS —— CAN —— None" could describe either ABS on "CAN —— None"
        # or "ABS —— CAN" with no bus. The exact MDF name "ABS —— None" wins.
        lab.write_text(
            f"[lab]\n{speed.name} —— None      \n{speed.name} —— CAN —— None\n{literal.name}\nABS —— CAN —— None\n"
        )
        self.widget.clear_filter()
        self.widget.load_filter_list(file_name=lab)
        with self.subTest("older export"):
            expected = dict.fromkeys(self.widget.mdf.whereis(speed.name), speed.name)
            expected[self.widget.mdf.whereis(literal.name)[0]] = literal.name
            self.assertEqual(self.widget._selected_filter, expected)

    def test_PushButton_ScrambleTexts(self):
        """
        Events:
            - Open 'FileWidget' with valid measurement.
            - Go to Tab: "Modify & Export": Index 1
            - Press PushButton "Scramble texts"
        Evaluate:
            - New file is created
            - No channel from first file is found in 2nd file (scrambled file)
        """
        # Event
        self.setUpFileWidget(measurement_file=self.measurement_file, default=True)
        channels = list(self.widget.mdf.channels_db)

        # Go to Tab: "Modify & Export": Index 1
        self.widget.aspects.setCurrentIndex(1)

        # Press PushButton ScrambleTexts
        QtTest.QTest.mouseClick(self.widget.scramble_btn, QtCore.Qt.MouseButton.LeftButton)
        self.processEvents(0.5)

        # Evaluate
        scrambled_filepath = pathlib.Path(self.test_workspace, "ASAP2_Demo_V171.scrambled.mf4")

        with OpenMDF(scrambled_filepath) as mdf_file:
            result = filter(lambda c: c in mdf_file.channels_db, channels)
            self.assertFalse(any(result))

    def test_ExportMDF(self):
        """
        When QThreads are running, event-loops needs to be processed.
        Events:
            - Open 'FileWidget' with valid measurement.
            - Go to Tab: "Modify & Export": Index 1
            - Set "channel_view" to "Natural sort"
            - Select two channels
            - Ensure that output format is MDF
            - Case 0:
                - Press PushButton Apply.
                    - Simulate that no valid path is provided.
            - Case 1:
                - Press PushButton Apply.
                    - Simulate that no valid path is provided.
        Evaluate:
            - Evaluate that file was created.
            - Open File and check that there are only two channels.
        """
        # Event
        self.setUpFileWidget(measurement_file=self.measurement_file, default=True)
        # Go to Tab: "Modify & Export": Index 1
        self.widget.aspects.setCurrentIndex(1)
        self.widget.filter_view.setCurrentText("Natural sort")

        count = 2
        selected_channels = []
        iterator = QtWidgets.QTreeWidgetItemIterator(self.widget.filter_tree)
        while iterator.value() and count:
            item = iterator.value()
            item.setCheckState(0, QtCore.Qt.CheckState.Checked)
            self.assertTrue(item.checkState(0) == QtCore.Qt.CheckState.Checked)
            selected_channels.append(item.text(0))
            iterator += 1
            count -= 1
        self.processEvents()

        # Evaluate that channels were added to "selected_filter_channels"
        for index in range(self.widget.selected_filter_channels.topLevelItemCount()):
            item = self.widget.selected_filter_channels.topLevelItem(index)
            self.assertIn(item.text(0), selected_channels)

        self.widget.output_format.setCurrentText("MDF")

        # Case 0:
        self.processEvents()
        with self.subTest("test_ExportMDF_0"):
            with (
                mock.patch("asammdf.gui.widgets.file.QtWidgets.QFileDialog.getSaveFileName") as mc_getSaveFileName,
                mock.patch("asammdf.gui.widgets.file.setup_progress") as mo_setup_progress,
            ):
                mc_getSaveFileName.return_value = None, None
                QtTest.QTest.mouseClick(self.widget.apply_btn, QtCore.Qt.MouseButton.LeftButton)
                self.processEvents()
            # Evaluate
            # Progress is not created
            mo_setup_progress.assert_not_called()

        # Case 1:
        self.processEvents()
        with self.subTest("test_ExportMDF_1"):
            saved_file = pathlib.Path(self.test_workspace, f"{self.id()}.mf4")
            with mock.patch("asammdf.gui.widgets.file.QtWidgets.QFileDialog.getSaveFileName") as mc_getSaveFileName:
                mc_getSaveFileName.return_value = str(saved_file), None
                QtTest.QTest.mouseClick(self.widget.apply_btn, QtCore.Qt.MouseButton.LeftButton)
                self.processEvents(1)
        # Wait for thread to finish
        self.processEvents(1)

        # Evaluate
        self.assertTrue(saved_file.exists())

        # TearDown Widget
        with OpenMDF(saved_file) as mdf_file:
            for sig in mdf_file.iter_channels():
                self.assertIn(sig.name, selected_channels)

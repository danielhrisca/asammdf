#!/usr/bin/env python
import pathlib
from unittest import mock

from PySide6 import QtCore, QtTest, QtWidgets

from asammdf.gui.dialogs.channel_group_info import ChannelGroupInfoDialog
from asammdf.gui.dialogs.channel_info import ChannelInfoDialog
from asammdf.gui.widgets.numeric import Numeric
from asammdf.gui.widgets.plot import Plot
from asammdf.gui.widgets.tabular import Tabular
from test.asammdf.gui.widgets.test_BaseFileWidget import TestFileWidget

# Note: If it's possible and make sense, use self.subTests
# to avoid initializing widgets multiple times and consume time.


class TestTabChannels(TestFileWidget):
    def tearDown(self):
        if self.mc_ErrorDialog.call_args:
            if (
                "<class 'RuntimeError'>" in self.mc_ErrorDialog.call_args.kwargs["message"]
                and self.mc_ErrorDialog.call_count == 1
            ):
                self.mc_ErrorDialog.reset_mock()
        self.mc_ErrorDialog.assert_not_called()
        self.mc_widget_ed.assert_not_called()
        super().tearDown()

    def test_load_file_with_metadata(self):
        """
        Events:
            - Open 'FileWidget' with valid measurement which has valid display file in metadata.
            - Store opened sub-windows typy, titles and channels names
            - Close all sub-windows
            - Press PushButton: "Load offline windows"
                - Simulate that dspf file from metadata was selected
        Expected:
            - Loaded sub-windows are the same as sub-windows opened with measurement file
        """

        def get_dspf_data(iterator):
            data = []
            for sub_win in iterator:
                w = sub_win.widget()
                if isinstance(w, Plot):
                    data.append(
                        {
                            "channels": [
                                w.channel_selection.topLevelItem(_).name
                                for _ in range(w.channel_selection.topLevelItemCount())
                            ],
                            "plot_bg_color": w.plot.backgroundBrush().color().name(),
                        }
                    )
                elif isinstance(w, Numeric):
                    data.append([sig.name for sig in w.channels.dataView.backend.signals])
                elif isinstance(w, Tabular):
                    data.append(list(w.tree.pgdf.df.columns))
                return data

        # Setup
        measurement_file = str(pathlib.Path(self.resource, "test_metadata.mf4"))
        # valid_dsp = str(pathlib.Path(self.resource, "valid.dsp"))

        # Event
        self.setUpFileWidget(measurement_file=measurement_file, default=True)

        # Prepare expected results
        dspf_file = self.widget.mdf.header._common_properties.get("pr_display_file", "")
        dspf_path = str(pathlib.Path(self.resource, dspf_file))

        sub_windows = self.get_sub_windows()
        titles = sorted(w.windowTitle() for w in self.widget.mdi_area.subWindowList())
        dspf_data = get_dspf_data(self.widget.mdi_area.subWindowList())

        # Close all sub-windows
        for w in self.widget.mdi_area.subWindowList():
            w.close()

        # Evaluate that all sub-windows was closed
        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 0)

        # Open DSPF
        with mock.patch("asammdf.gui.widgets.file.QtWidgets.QFileDialog.getOpenFileName") as mo_getOpenFileName:
            mo_getOpenFileName.return_value = dspf_path, None
            QtTest.QTest.mouseClick(
                self.widget.load_channel_list_btn,
                QtCore.Qt.MouseButton.LeftButton,
            )
            self.processEvents()

            # Evaluate
            self.assertListEqual(sub_windows, self.get_sub_windows())
            self.assertListEqual(titles, sorted(w.windowTitle() for w in self.widget.mdi_area.subWindowList()))
            for old, new in zip(dspf_data, get_dspf_data(self.widget.mdi_area.subWindowList()), strict=False):
                if isinstance(old, dict):
                    self.assertListEqual(new["channels"], old["channels"])
                    self.assertEqual(new["plot_bg_color"], old["plot_bg_color"])
                elif isinstance(old, list):
                    self.assertListEqual(old, new)

    def test_PushButton_LoadOfflineWindows_DSP(self):
        """
        Events:
            - Open 'FileWidget' with valid measurement.
            - Press PushButton: "Load offline windows"
                - Simulate that valid file was selected
        Expected:
            - Evaluate that Plot Window is added in "mdi_area"
        """
        # Setup
        measurement_file = str(pathlib.Path(self.resource, "ASAP2_Demo_V171.mf4"))
        valid_dsp = str(pathlib.Path(self.resource, "valid.dsp"))
        # Event
        self.setUpFileWidget(measurement_file=measurement_file, default=True)

        with mock.patch("asammdf.gui.widgets.file.QtWidgets.QFileDialog.getOpenFileName") as mo_getOpenFileName:
            mo_getOpenFileName.return_value = valid_dsp, None
            QtTest.QTest.mouseClick(
                self.widget.load_channel_list_btn,
                QtCore.Qt.MouseButton.LeftButton,
            )
            # Evaluate
            self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)
            widget_types = self.get_sub_windows()
            self.assertIn("Plot", widget_types)

    @mock.patch("asammdf.gui.widgets.file.ErrorDialog")
    def test_PushButton_LoadOfflineWindows_DSPF(self, mc_file_ErrorDialog):
        """
        Events:
            - Open 'FileWidget' with valid measurement.
            - Case 0:
                - Press PushButton: "Load offline windows"
                    - Simulate that no file was selected
            - Case 1:
                - Press PushButton: "Load offline windows"
                    - Simulate that file with non-supported extension was selected
            - Case 2:
                - Press PushButton: "Load offline windows"
                    - Simulate that invalid "dspf" file was selected: json decode error
            - Case 3:
                - Press PushButton: "Load offline windows"
                    - Simulate that invalid "dspf" file was selected: Plot Section Key Error
            - Case 4:
                - Press PushButton: "Load offline windows"
                    - Simulate that invalid "dspf" file was selected: Numeric Section Key Error
            - Case 5:
                - Press PushButton: "Load offline windows"
                    - Simulate that invalid "dspf" file was selected: Tabular Section Key Error
            - Case 6:
                - Press PushButton: "Load offline windows"
                    - Simulate that valid "dspf" file was selected
        Expected:
            - Case 0: No dpsf file was selected
                - Evaluate that method "load_window" is not called
            - Case 1: File with non-supported extension was selected
                - Evaluate that method "load_window" is not called
            - Case 2: Json Decode Error
                - Evaluate that method "load_window" is not called
                - Evaluate that "ErrorDialog" widget is called
            - Case 3: Plot Section Key Error
                - Evaluate that there is no Plot window in "mdi_area"
                - Evaluate that "ErrorDialog" widget is called
            - Case 4: Numeric Section Key Error
                - Evaluate that there is no Numeric window in "mdi_area"
                - Evaluate that "ErrorDialog" widget is called
            - Case 5: Tabular Section Key Error
                - Evaluate that there is no Tabuler window in "mdi_area"
                - Evaluate that "ErrorDialog" widget is called
            - Case 6:
                - Evaluate that there 3 sub-windows in "mdi_area" when measurement file is loaded
                    - Numeric, Plot, Tabular
        """
        # Setup
        measurement_file = str(pathlib.Path(self.resource, "ASAP2_Demo_V171.mf4"))
        valid_dspf = str(pathlib.Path(self.resource, "valid.dspf"))
        invalid_json_decode_error_dspf = str(pathlib.Path(self.resource, "invalid_JsonDecodeError.dspf"))
        invalid_numeric_section_key_error_dspf = str(pathlib.Path(self.resource, "invalid_NumericSectionKeyError.dspf"))
        invalid_plot_section_key_error_dspf = str(pathlib.Path(self.resource, "invalid_PlotSectionKeyError.dspf"))
        invalid_tabular_section_key_error_dspf = str(pathlib.Path(self.resource, "invalid_TabularSectionKeyError.dspf"))

        # Event
        self.setUpFileWidget(measurement_file=measurement_file, default=True)

        # Case 0
        with self.subTest("test_PushButton_LoadOfflineWindows_DSPF_0"):
            with (
                mock.patch.object(self.widget, "load_window", wraps=self.widget.load_window) as mo_load_window,
                mock.patch("asammdf.gui.widgets.file.QtWidgets.QFileDialog.getOpenFileName") as mo_getOpenFileName,
            ):
                mo_getOpenFileName.return_value = None, None
                QtTest.QTest.mouseClick(
                    self.widget.load_channel_list_btn,
                    QtCore.Qt.MouseButton.LeftButton,
                )
                # Evaluate
                mo_load_window.assert_not_called()
                self.assertEqual(len(self.widget.mdi_area.subWindowList()), 0)

        # Case 1
        with self.subTest("test_PushButton_LoadOfflineWindows_DSPF_1"):
            with (
                mock.patch.object(self.widget, "load_window", wraps=self.widget.load_window) as mo_load_window,
                mock.patch("asammdf.gui.widgets.file.QtWidgets.QFileDialog.getOpenFileName") as mo_getOpenFileName,
            ):
                mo_getOpenFileName.return_value = valid_dspf[:-2], None
                QtTest.QTest.mouseClick(
                    self.widget.load_channel_list_btn,
                    QtCore.Qt.MouseButton.LeftButton,
                )
                # Evaluate
                mo_load_window.assert_not_called()
                self.assertEqual(len(self.widget.mdi_area.subWindowList()), 0)

        # Case 2
        with self.subTest("test_PushButton_LoadOfflineWindows_DSPF_2"):
            with (
                mock.patch.object(self.widget, "load_window", wraps=self.widget.load_window) as mo_load_window,
                mock.patch("asammdf.gui.widgets.file.QtWidgets.QFileDialog.getOpenFileName") as mo_getOpenFileName,
            ):
                mo_getOpenFileName.return_value = (
                    invalid_json_decode_error_dspf,
                    None,
                )
                QtTest.QTest.mouseClick(
                    self.widget.load_channel_list_btn,
                    QtCore.Qt.MouseButton.LeftButton,
                )
                # Evaluate
                mo_load_window.assert_not_called()
                self.assertEqual(len(self.widget.mdi_area.subWindowList()), 0)
                self.mc_ErrorDialog.assert_called()
                self.mc_ErrorDialog.reset_mock()

        # Case 3
        with self.subTest("test_PushButton_LoadOfflineWindows_DSPF_3"):
            with (
                mock.patch("asammdf.gui.widgets.file.ErrorDialog") as mc_ErrorDialog,
                mock.patch("asammdf.gui.widgets.file.QtWidgets.QFileDialog.getOpenFileName") as mo_getOpenFileName,
            ):
                mo_getOpenFileName.return_value = (
                    invalid_plot_section_key_error_dspf,
                    None,
                )
                QtTest.QTest.mouseClick(
                    self.widget.load_channel_list_btn,
                    QtCore.Qt.MouseButton.LeftButton,
                )
                # Evaluate
                self.assertEqual(len(self.widget.mdi_area.subWindowList()), 2)
                mc_ErrorDialog.assert_called()
                mc_ErrorDialog.reset_mock()
                widget_types = self.get_sub_windows()
                self.assertNotIn("Plot", widget_types)

        # Case 4
        with self.subTest("test_PushButton_LoadOfflineWindows_DSPF_4"):
            with (
                mock.patch("asammdf.gui.widgets.file.ErrorDialog") as mc_ErrorDialog,
                mock.patch("asammdf.gui.widgets.file.QtWidgets.QFileDialog.getOpenFileName") as mo_getOpenFileName,
            ):
                mo_getOpenFileName.return_value = (
                    invalid_numeric_section_key_error_dspf,
                    None,
                )
                QtTest.QTest.mouseClick(
                    self.widget.load_channel_list_btn,
                    QtCore.Qt.MouseButton.LeftButton,
                )
                # Evaluate
                self.assertEqual(len(self.widget.mdi_area.subWindowList()), 2)
                mc_ErrorDialog.assert_called()
                mc_ErrorDialog.reset_mock()
                widget_types = self.get_sub_windows()
                self.assertNotIn("Numeric", widget_types)

        # Case 5
        with self.subTest("test_PushButton_LoadOfflineWindows_DSPF_5"):
            with (
                mock.patch("asammdf.gui.widgets.file.ErrorDialog") as mc_ErrorDialog,
                mock.patch("asammdf.gui.widgets.file.QtWidgets.QFileDialog.getOpenFileName") as mo_getOpenFileName,
            ):
                mo_getOpenFileName.return_value = (
                    invalid_tabular_section_key_error_dspf,
                    None,
                )
                QtTest.QTest.mouseClick(
                    self.widget.load_channel_list_btn,
                    QtCore.Qt.MouseButton.LeftButton,
                )
                # Evaluate
                self.assertEqual(len(self.widget.mdi_area.subWindowList()), 2)
                mc_ErrorDialog.assert_called()
                mc_ErrorDialog.reset_mock()
                widget_types = self.get_sub_windows()
                self.assertNotIn("Tabular", widget_types)

        # Case 6
        with self.subTest("test_PushButton_LoadOfflineWindows_DSPF_6"):
            with (
                mock.patch.object(self.widget, "load_window", wraps=self.widget.load_window) as mo_load_window,
                mock.patch("asammdf.gui.widgets.file.QtWidgets.QFileDialog.getOpenFileName") as mo_getOpenFileName,
            ):
                mo_getOpenFileName.return_value = valid_dspf, None
                QtTest.QTest.mouseClick(
                    self.widget.load_channel_list_btn,
                    QtCore.Qt.MouseButton.LeftButton,
                )
                # Evaluate
                mo_load_window.assert_called()
                self.assertEqual(len(self.widget.mdi_area.subWindowList()), 3)
                widget_types = self.get_sub_windows()
                self.assertListEqual(["Numeric", "Plot", "Tabular"], widget_types)

        mc_file_ErrorDialog.assert_not_called()

    def test_PushButton_LoadOfflineWindows_LAB(self):
        """
        Events:
            - Open 'FileWidget' with valid measurement.
            - Ensure that Channels View is set to "Internal file structure"
            - Case 0:
                - Press PushButton: "Load offline windows"
                    - Simulate that "lab" file with empty section was selected
            - Case 1:
                - Press PushButton: "Load offline windows"
                    - Simulate that "lab" with no sections was provided
            - Case 2:
                - Press PushButton: "Load offline windows"
                    - Simulate that valid "lab" file was selected
        Expected:
            - Case 0 & 1:
                - Evaluate that no channel is checked
            - Case 2:
                - Evaluate that 7 channels are checked
        """
        # Setup
        valid_lab = str(pathlib.Path(self.resource, "valid.lab"))
        invalid_missing_section_lab = str(pathlib.Path(self.resource, "invalid_MissingSection.lab"))
        invalid_empty_section_lab = str(pathlib.Path(self.resource, "invalid_EmptySection.lab"))
        measurement_file = str(pathlib.Path(self.resource, "ASAP2_Demo_V171.mf4"))

        # Event
        self.setUpFileWidget(measurement_file=measurement_file, default=True)
        # Switch ComboBox to "Internal file structure"
        self.widget.channel_view.setCurrentText("Internal file structure")
        # Case 0:
        with self.subTest("test_PushButton_LoadOfflineWindows_LAB_0"):
            with mock.patch("asammdf.gui.widgets.file.QtWidgets.QFileDialog.getOpenFileName") as mo_getOpenFileName:
                mo_getOpenFileName.return_value = (
                    invalid_empty_section_lab,
                    None,
                )
                QtTest.QTest.mouseClick(
                    self.widget.load_channel_list_btn,
                    QtCore.Qt.MouseButton.LeftButton,
                )
                # Evaluate
                self.assertEqual(len(self.widget.mdi_area.subWindowList()), 0)

                checked_items = []
                iterator = QtWidgets.QTreeWidgetItemIterator(self.widget.channels_tree)
                while iterator.value():
                    item = iterator.value()
                    if item.parent() is None:
                        iterator += 1
                        continue
                    if item.checkState(0) == QtCore.Qt.CheckState.Checked:
                        checked_items.append(item.text(0))
                    iterator += 1
                self.assertEqual(0, len(checked_items))
                self.assertNotRegex(
                    str(self.mc_ErrorDialog.mock_calls),
                    r"local variable .* referenced before assignment",
                )

        # Case 1:
        with self.subTest("test_PushButton_LoadOfflineWindows_LAB_1"):
            with mock.patch("asammdf.gui.widgets.file.QtWidgets.QFileDialog.getOpenFileName") as mo_getOpenFileName:
                mo_getOpenFileName.return_value = (
                    invalid_missing_section_lab,
                    None,
                )
                QtTest.QTest.mouseClick(
                    self.widget.load_channel_list_btn,
                    QtCore.Qt.MouseButton.LeftButton,
                )
                # Evaluate
                self.assertEqual(len(self.widget.mdi_area.subWindowList()), 0)

                checked_items = []
                iterator = QtWidgets.QTreeWidgetItemIterator(self.widget.channels_tree)
                while iterator.value():
                    item = iterator.value()
                    if item.parent() is None:
                        iterator += 1
                        continue
                    if item.checkState(0) == QtCore.Qt.CheckState.Checked:
                        checked_items.append(item.text(0))
                    iterator += 1
                self.assertEqual(0, len(checked_items))
                self.assertNotRegex(
                    str(self.mc_ErrorDialog.mock_calls),
                    r"local variable .* referenced before assignment",
                )

        # Case 2:
        with self.subTest("test_PushButton_LoadOfflineWindows_LAB_2"):
            with (
                mock.patch("asammdf.gui.widgets.file.QtWidgets.QFileDialog.getOpenFileName") as mo_getOpenFileName,
                mock.patch("asammdf.gui.widgets.file.QtWidgets.QInputDialog.getItem") as mo_getItem,
            ):
                mo_getOpenFileName.return_value = valid_lab, None
                mo_getItem.return_value = "lab", True
                QtTest.QTest.mouseClick(
                    self.widget.load_channel_list_btn,
                    QtCore.Qt.MouseButton.LeftButton,
                )
                # Evaluate
                self.assertEqual(len(self.widget.mdi_area.subWindowList()), 0)

                checked_items = []
                iterator = QtWidgets.QTreeWidgetItemIterator(self.widget.channels_tree)
                while iterator.value():
                    item = iterator.value()
                    if item.parent() is None:
                        iterator += 1
                        continue
                    if item.checkState(0) == QtCore.Qt.CheckState.Checked:
                        checked_items.append(item.text(0))
                    iterator += 1
                self.assertEqual(7, len(checked_items))
                self.assertSetEqual(
                    {
                        "ASAM.M.SCALAR.FLOAT64.IDENTICAL",
                        "ASAM.M.SCALAR.UBYTE.FORM_X_PLUS_4",
                        "ASAM_[1].M.MATRIX_DIM_16.UBYTE.IDENTICAL",
                        "ASAM_[11].M.MATRIX_DIM_16.UBYTE.IDENTICAL",
                        "ASAM_[15].M.MATRIX_DIM_16.UBYTE.IDENTICAL",
                        "time",
                    },
                    set(checked_items),
                )

    def test_PushButton_SaveOfflineWindows(self):
        """

        Events:
            - Open 'FileWidget' with valid measurement.
            - Ensure that Channels View is set to "Internal file structure"
            - Press PushButton: "Load offline windows"
                - Simulate that valid "dspf" file was selected
            - Close all Numeric and Tabular windows
            - Add new Plot window with few channels
            - Press PushButton: "Save offline windows"
            - Close all windows.
            - Press PushButton: "Load offline windows"
                - Simulate that saved valid "dspf" file was selected
        Evaluate:
            - Evaluate that new dspf file was saved.
            - Evaluate that two Plot Windows are loaded.
        """
        # Setup
        valid_dspf = str(pathlib.Path(self.resource, "valid.dspf"))
        saved_dspf = pathlib.Path(self.test_workspace, f"{self.id()}.dspf")
        measurement_file = str(pathlib.Path(self.resource, "ASAP2_Demo_V171.mf4"))

        self.setUpFileWidget(measurement_file=measurement_file, default=True)
        # Switch ComboBox to "Internal file structure"
        self.widget.channel_view.setCurrentText("Internal file structure")

        # Load Display File
        self.load_display_file(display_file=valid_dspf)
        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 3)

        # Close all Numeric and Tabular windows
        for window in self.widget.mdi_area.subWindowList():
            if window.widget().__class__.__name__ in ("Numeric", "Tabular"):
                window.close()

        # Drag Part

        # Add a new Plot window with few channels
        self.create_window(window_type="Plot", channels_names=["ASAM.M.SCALAR.UBYTE.VTAB_RANGE_NO_DEFAULT_VALUE"])

        # Press PushButton: "Save offline windows"
        with mock.patch("asammdf.gui.widgets.file.QtWidgets.QFileDialog.getSaveFileName") as mo_getSaveFileName:
            mo_getSaveFileName.return_value = str(saved_dspf), None
            QtTest.QTest.mouseClick(
                self.widget.save_channel_list_btn,
                QtCore.Qt.MouseButton.LeftButton,
            )
        # Evaluate
        self.assertTrue(saved_dspf.exists())

        # Event
        self.load_display_file(display_file=saved_dspf)
        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 2)
        widget_types = {w.widget().__class__.__name__ for w in self.widget.mdi_area.subWindowList()}
        self.assertSetEqual({"Plot"}, widget_types)

    def test_PushButton_SelectAll(self):
        """
        Events:
            - Open 'FileWidget' with valid measurement.
            - Switch ComboBox to "Natural sort"
            - Press PushButton: "Select all the channels"
            - Clear selection
            - Switch ComboBox to "Internal file structure"
            - Press PushButton: "Select all the channels"
            - Switch ComboBox to "Selected channels only"
        Evaluate:
            - Evaluate that all channels from "channels_tree" are checked.
        """
        measurement_file = str(pathlib.Path(self.resource, "ASAP2_Demo_V171.mf4"))

        # Event
        self.setUpFileWidget(measurement_file=measurement_file, default=True)

        # Switch ComboBox to "Natural sort"
        self.widget.channel_view.setCurrentText("Natural Sort")
        # Press PushButton: "Select all the channels"
        QtTest.QTest.mouseClick(self.widget.select_all_btn, QtCore.Qt.MouseButton.LeftButton)

        # Evaluate
        iterator = QtWidgets.QTreeWidgetItemIterator(self.widget.channels_tree)
        while iterator.value():
            item = iterator.value()
            self.assertTrue(item.checkState(0))
            iterator += 1

        # Clear all
        iterator = QtWidgets.QTreeWidgetItemIterator(self.widget.channels_tree)
        while iterator.value():
            item = iterator.value()
            item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
            self.assertEqual(item.checkState(0), QtCore.Qt.CheckState.Unchecked)
            iterator += 1

        # Switch ComboBox to "Internal file structure"
        self.widget.channel_view.setCurrentText("Internal file structure")
        # Press PushButton: "Select all the channels"
        QtTest.QTest.mouseClick(self.widget.select_all_btn, QtCore.Qt.MouseButton.LeftButton)

        # Evaluate
        iterator = QtWidgets.QTreeWidgetItemIterator(self.widget.channels_tree)
        while iterator.value():
            item = iterator.value()
            self.assertTrue(item.checkState(0))
            iterator += 1

        # Switch ComboBox to "Selected channels only"
        self.widget.channel_view.setCurrentText("Selected channels only")

        # Evaluate
        iterator = QtWidgets.QTreeWidgetItemIterator(self.widget.channels_tree)
        while iterator.value():
            item = iterator.value()
            self.assertTrue(item.checkState(0))
            iterator += 1

    def test_PushButton_ClearAll(self):
        """
        Events:
            - Open 'FileWidget' with valid measurement.
            - Switch ComboBox to "Natural sort"
            - Select all channels
            - Press PushButton: "Clear all selected channels"
            - Switch ComboBox to "Internal file structure"
            - Select all channels
            - Press PushButton: "Clear all selected channels"
            - Switch ComboBox to "Selected channels only"
        Evaluate:
            - Evaluate that all channels from "channels_tree" are unchecked.
        """
        measurement_file = str(pathlib.Path(self.resource, "ASAP2_Demo_V171.mf4"))

        # Event
        self.setUpFileWidget(measurement_file=measurement_file, default=True)

        # Switch ComboBox to "Natural sort"
        self.widget.channel_view.setCurrentText("Natural Sort")
        # Select all
        iterator = QtWidgets.QTreeWidgetItemIterator(self.widget.channels_tree)
        while iterator.value():
            item = iterator.value()
            item.setCheckState(0, QtCore.Qt.CheckState.Checked)
            self.assertEqual(QtCore.Qt.CheckState.Checked, item.checkState(0))
            iterator += 1
        # Press PushButton: "Clear all selected channels"
        QtTest.QTest.mouseClick(self.widget.clear_channels_btn, QtCore.Qt.MouseButton.LeftButton)

        # Evaluate
        iterator = QtWidgets.QTreeWidgetItemIterator(self.widget.channels_tree)
        while iterator.value():
            item = iterator.value()
            self.assertEqual(QtCore.Qt.CheckState.Unchecked, item.checkState(0))
            iterator += 1

        # Switch ComboBox to "Internal file structure"
        self.widget.channel_view.setCurrentText("Internal file structure")
        while iterator.value():
            item = iterator.value()
            item.setCheckState(0, QtCore.Qt.CheckState.Checked)
            self.assertEqual(QtCore.Qt.CheckState.Checked, item.checkState(0))
            iterator += 1
        # Press PushButton: "Clear all selected channels"
        QtTest.QTest.mouseClick(self.widget.clear_channels_btn, QtCore.Qt.MouseButton.LeftButton)

        # Evaluate
        iterator = QtWidgets.QTreeWidgetItemIterator(self.widget.channels_tree)
        while iterator.value():
            item = iterator.value()
            self.assertEqual(QtCore.Qt.CheckState.Unchecked, item.checkState(0))
            iterator += 1

        # Switch ComboBox to "Selected channels only"
        self.widget.channel_view.setCurrentText("Selected channels only")

        # Evaluate
        iterator = QtWidgets.QTreeWidgetItemIterator(self.widget.channels_tree)
        while iterator.value():
            item = iterator.value()
            self.assertEqual(QtCore.Qt.CheckState.Unchecked, item.checkState(0))
            iterator += 1

    def test_PushButton_Search(self):
        """
        Events:
            - Open 'FileWidget' with valid measurement.
            - Case 0:
                - Press PushButton: "Search and select channels"
                    - Simulate that "AdvancedSearch" was closed/cancelled (no signal was selected)
            - Case 1:
                - Press PushButton: "Search and select channels"
                    - Simulate that 2 channels were selected and PushButton "Check channels" was pressed.
            - Case 2:
                - Clear 'channel_tree' selection
                - Press PushButton: "Search and select channels"
                    - Simulate that 2 channels were selected and PushButton "Add Window" was pressed.
        Evaluate:
            - Case 0:
                - Evaluate that no channel is selected in "channel_tree"
            - Case 1:
                - Evaluate that 2 channels are selected in "channel_tree"
            - Case 2:
                - Evaluate that 2 channels are selected in "channel_tree"
                - Plot Window is added.
        """
        # Setup
        measurement_file = str(pathlib.Path(self.resource, "ASAP2_Demo_V171.mf4"))
        # Event
        self.setUpFileWidget(measurement_file=measurement_file, default=True)
        # Case 0:
        with self.subTest("test_PushButton_Search_0"):
            with mock.patch("asammdf.gui.widgets.file.AdvancedSearch") as mc_AdvancedSearch:
                mc_AdvancedSearch.return_value.result = {}
                mc_AdvancedSearch.return_value.pattern_window = False
                mc_AdvancedSearch.return_value.add_window_request = False

                # - Press PushButton: "Search and select channels"
                QtTest.QTest.mouseClick(self.widget.advanced_search_btn, QtCore.Qt.MouseButton.LeftButton)
                # Evaluate
                iterator = QtWidgets.QTreeWidgetItemIterator(self.widget.channels_tree)
                while iterator.value():
                    item = iterator.value()
                    self.assertEqual(item.checkState(0), QtCore.Qt.CheckState.Unchecked)
                    iterator += 1

        # Case 1:
        with self.subTest("test_PushButton_Search_1"):
            with mock.patch("asammdf.gui.widgets.file.AdvancedSearch") as mc_AdvancedSearch:
                mc_AdvancedSearch.return_value.result = {
                    (4, 3): "ASAM.M.SCALAR.FLOAT64.IDENTICAL",
                    (2, 10): "ASAM.M.SCALAR.FLOAT32.IDENTICAL",
                }
                mc_AdvancedSearch.return_value.pattern_window = False
                mc_AdvancedSearch.return_value.add_window_request = False

                # - Press PushButton: "Search and select channels"
                QtTest.QTest.mouseClick(self.widget.advanced_search_btn, QtCore.Qt.MouseButton.LeftButton)
                # Evaluate
                checked_channels = 0
                iterator = QtWidgets.QTreeWidgetItemIterator(self.widget.channels_tree)
                while iterator.value():
                    item = iterator.value()
                    if item.checkState(0) == QtCore.Qt.CheckState.Checked:
                        checked_channels += 1
                    iterator += 1
                self.assertEqual(2, checked_channels)
                self.assertEqual(len(self.widget.mdi_area.subWindowList()), 0)

        # Case 2:
        with self.subTest("test_PushButton_Search_2"):
            # - Clear 'channel_tree' selection
            iterator = QtWidgets.QTreeWidgetItemIterator(self.widget.channels_tree)
            while iterator.value():
                item = iterator.value()
                item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
                iterator += 1
            with (
                mock.patch("asammdf.gui.widgets.file.AdvancedSearch") as mc_AdvancedSearch,
                mock.patch("asammdf.gui.widgets.file.WindowSelectionDialog") as mc_WindowSelectionDialog,
            ):
                mc_AdvancedSearch.return_value.result = {
                    (4, 3): "ASAM.M.SCALAR.FLOAT64.IDENTICAL",
                    (2, 10): "ASAM.M.SCALAR.FLOAT32.IDENTICAL",
                }
                mc_AdvancedSearch.return_value.pattern_window = False
                mc_AdvancedSearch.return_value.add_window_request = True
                mc_WindowSelectionDialog.return_value.result.return_value = True
                mc_WindowSelectionDialog.return_value.selected_type.return_value = "New plot window"
                mc_WindowSelectionDialog.return_value.disable_new_channels.return_value = False

                # - Press PushButton: "Search and select channels"
                QtTest.QTest.mouseClick(self.widget.advanced_search_btn, QtCore.Qt.MouseButton.LeftButton)
                # Evaluate
                checked_channels = 0
                iterator = QtWidgets.QTreeWidgetItemIterator(self.widget.channels_tree)
                while iterator.value():
                    item = iterator.value()
                    if item.checkState(0) == QtCore.Qt.CheckState.Checked:
                        checked_channels += 1
                    iterator += 1
                self.assertEqual(2, checked_channels)
                self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)
                widget_types = self.get_sub_windows()
                self.assertIn("Plot", widget_types)

    def test_PushButton_CreateWindow(self):
        """
        Events:
            - Open 'FileWidget' with valid measurement.
            - Switch ComboBox to "Natural sort"
            - Case 0:
                - Press PushButton "Create Window"
                    - Simulate that "WindowSelectionDialog" is cancelled/closed.
            - Case 1:
                - Press PushButton "Create Window"
                    - Simulate that Window Type is Plot.
            - Case 2:
                - Select one channel
                - Press PushButton "Create Window"
                    - Simulate that Window Type is Numeric.
            - Case 3:
                - Press PushButton "Create Window"
                    - Simulate that Window Type is Tabular.
            - Case 4:
                - Press PushButton "Create Window"
                    - Simulate that Window Type is Plot.
        Evaluate:
            - Evaluate that 4 sub-windows are created.
            - Check if windows contain channel selected.
        """
        # Setup
        measurement_file = str(pathlib.Path(self.resource, "ASAP2_Demo_V171.mf4"))
        # Event
        self.setUpFileWidget(measurement_file=measurement_file, default=True)
        self.widget.channel_view.setCurrentText("Natural sort")

        # Case 0:
        with self.subTest("test_PushButton_CreateWindow_0"):
            with mock.patch("asammdf.gui.widgets.file.WindowSelectionDialog") as mc_WindowSelectionDialog:
                mc_WindowSelectionDialog.return_value.result.return_value = False
                # - Press PushButton "Create Window"
                QtTest.QTest.mouseClick(self.widget.create_window_btn, QtCore.Qt.MouseButton.LeftButton)
            # Evaluate
            self.assertEqual(len(self.widget.mdi_area.subWindowList()), 0)

        # Case 1:
        with self.subTest("test_PushButton_CreateWindow_1"):
            with mock.patch("asammdf.gui.widgets.file.WindowSelectionDialog") as mc_WindowSelectionDialog:
                mc_WindowSelectionDialog.return_value.result.return_value = True
                mc_WindowSelectionDialog.return_value.selected_type.return_value = "Plot"
                # - Press PushButton "Create Window"
                QtTest.QTest.mouseClick(self.widget.create_window_btn, QtCore.Qt.MouseButton.LeftButton)
            # Evaluate
            self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)
            widget_types = self.get_sub_windows()
            self.assertIn("Plot", widget_types)

        # Case 2:
        with self.subTest("test_PushButton_CreateWindow_2"):
            with mock.patch("asammdf.gui.widgets.file.WindowSelectionDialog") as mc_WindowSelectionDialog:
                mc_WindowSelectionDialog.return_value.result.return_value = True
                mc_WindowSelectionDialog.return_value.selected_type.return_value = "Numeric"
                # - Select one channel
                channel = self.widget.channels_tree.topLevelItem(0).text(0)
                self.widget.channels_tree.topLevelItem(0).setCheckState(0, QtCore.Qt.CheckState.Checked)
                # - Press PushButton "Create Window"
                QtTest.QTest.mouseClick(self.widget.create_window_btn, QtCore.Qt.MouseButton.LeftButton)
            # Evaluate
            self.assertEqual(len(self.widget.mdi_area.subWindowList()), 2)
            widget_types = self.get_sub_windows()
            self.assertIn("Numeric", widget_types)
            numeric_data = self.widget.mdi_area.subWindowList()[1].widget().channels.dataView
            numeric_channel = numeric_data.model().data(numeric_data.model().index(0, 0))
            self.assertEqual(channel, numeric_channel)

        # Case 3:
        with self.subTest("test_PushButton_CreateWindow_3"):
            with mock.patch("asammdf.gui.widgets.file.WindowSelectionDialog") as mc_WindowSelectionDialog:
                mc_WindowSelectionDialog.return_value.result.return_value = True
                mc_WindowSelectionDialog.return_value.selected_type.return_value = "Tabular"
                # - Press PushButton "Create Window"
                QtTest.QTest.mouseClick(self.widget.create_window_btn, QtCore.Qt.MouseButton.LeftButton)
            # Evaluate
            self.assertEqual(len(self.widget.mdi_area.subWindowList()), 3)
            widget_types = self.get_sub_windows()
            self.assertIn("Tabular", widget_types)

        # Case 4:
        with self.subTest("test_PushButton_CreateWindow_4"):
            with mock.patch("asammdf.gui.widgets.file.WindowSelectionDialog") as mc_WindowSelectionDialog:
                mc_WindowSelectionDialog.return_value.result.return_value = True
                mc_WindowSelectionDialog.return_value.selected_type.return_value = "Plot"
                # - Press PushButton "Create Window"
                QtTest.QTest.mouseClick(self.widget.create_window_btn, QtCore.Qt.MouseButton.LeftButton)
            # Evaluate
            self.assertEqual(len(self.widget.mdi_area.subWindowList()), 4)
            widget_types = self.get_sub_windows()
            self.assertIn("Plot", widget_types)
            plot_widget = self.widget.mdi_area.subWindowList()[3].widget()
            plot_channel = plot_widget.channel_selection.topLevelItem(0).text(0)
            self.assertEqual(channel, plot_channel)

    def test_DoubleClick_Channel(self):
        """
        Events:
            - Open 'FileWidget' with valid measurement.
            - Ensure that "channel_view" is set to "Internal file structure"
            - Case 0:
                - DoubleClick on Channel Group
            - Case 1:
                - DoubleClick on Channel
        Evaluate:
            - Evaluate that new dialog is visible and display channel meta-data.
        """
        # Setup
        measurement_file = str(pathlib.Path(self.resource, "ASAP2_Demo_V171.mf4"))
        # Event
        self.setUpFileWidget(measurement_file=measurement_file, default=True)
        self.widget.channel_view.setCurrentText("Internal file structure")

        first_item = self.widget.channels_tree.topLevelItem(0)
        first_item_center = self.widget.channels_tree.visualItemRect(first_item).center()
        QtTest.QTest.mouseClick(
            self.widget.channels_tree.viewport(),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
            first_item_center,
        )
        self.processEvents()
        # Case 0:
        with mock.patch(
            "asammdf.gui.widgets.file.ChannelGroupInfoDialog",
            wraps=ChannelGroupInfoDialog,
        ) as mc_ChannelGroupInfoDialog:
            QtTest.QTest.mouseDClick(
                self.widget.channels_tree.viewport(),
                QtCore.Qt.MouseButton.LeftButton,
                QtCore.Qt.KeyboardModifier.NoModifier,
                first_item_center,
            )
            # Evaluate
            mc_ChannelGroupInfoDialog.assert_called()
            # Identify child: ChannelGroupInfoDialog
            for child in self.widget.children():
                if child.__class__.__name__ == "ChannelGroupInfoDialog":
                    break
            if hasattr(child, "isVisible") and hasattr(child, "close"):
                self.assertTrue(child.isVisible())
                child.close()
                self.assertFalse(child.isVisible())

        # Case 1:
        child_item = first_item.child(0)
        child_item_center = self.widget.channels_tree.visualItemRect(child_item).center()
        QtTest.QTest.mouseClick(
            self.widget.channels_tree.viewport(),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
            child_item_center,
        )
        self.processEvents()
        with mock.patch(
            "asammdf.gui.widgets.file.ChannelInfoDialog",
            wraps=ChannelInfoDialog,
        ) as mc_ChannelGroupInfoDialog:
            QtTest.QTest.mouseDClick(
                self.widget.channels_tree.viewport(),
                QtCore.Qt.MouseButton.LeftButton,
                QtCore.Qt.KeyboardModifier.NoModifier,
                child_item_center,
            )
            # Evaluate
            mc_ChannelGroupInfoDialog.assert_called()
            # Identify child: ChannelInfoDialog
            for child in self.widget.children():
                if child.__class__.__name__ == "ChannelInfoDialog":
                    break
            if hasattr(child, "isVisible") and hasattr(child, "close"):
                self.assertTrue(child.isVisible())
                child.close()
                self.assertFalse(child.isVisible())

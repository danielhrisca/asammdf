#!/usr/bin/env python
import pathlib
import sys
from test.asammdf.gui import QtCore, QtTest, QtWidgets
from test.asammdf.gui.test_base import TestBase
import time
import unittest
from unittest import mock

from PySide6 import QtCore, QtWidgets

from asammdf.gui.widgets.file import FileWidget


class DragAndDrop:
    class MoveThread(QtCore.QThread):
        def __init__(self, widget, position=None, step=None, drop=False):
            super().__init__()
            self.widget = widget
            self.position = position
            self.step = step
            self.drop = drop

        def run(self):
            time.sleep(0.1)
            if not self.step:
                QtTest.QTest.mouseMove(self.widget, self.position)
            else:
                for step in range(self.step):
                    QtTest.QTest.mouseMove(
                        self.widget, self.position + QtCore.QPoint(step, step)
                    )
                    QtTest.QTest.qWait(2)
            QtTest.QTest.qWait(50)
            if self.drop:
                QtTest.QTest.mouseRelease(
                    self.widget,
                    QtCore.Qt.LeftButton,
                    QtCore.Qt.NoModifier,
                    self.position,
                )
                QtTest.QTest.qWait(10)

    def __init__(self, source_widget, destination_widget, source_pos, destination_pos):
        # Press on Source Widget
        QtTest.QTest.mousePress(
            source_widget, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, source_pos
        )
        # Drag few pixels in order to detect startDrag event
        # drag_thread = DragAndDrop.MoveThread(widget=source_widget, position=source_pos, step=50)
        # drag_thread.start()
        # Move to Destination Widget
        move_thread = DragAndDrop.MoveThread(
            widget=destination_widget, position=destination_pos, drop=True
        )
        move_thread.start()

        source_widget.startDrag(QtCore.Qt.MoveAction)
        QtTest.QTest.qWait(50)

        # drag_thread.wait()
        move_thread.wait()
        move_thread.quit()


@unittest.skipIf(
    sys.platform == "darwin", "Test Development on MacOS was not done yet."
)
class TestFileWidget(TestBase):
    # Note: If it's possible and make sense use self.subTests
    # to avoid initialize widgets multiple times and consume time.
    def setUp(self):
        super().setUp()
        self.widget = None

    def tearDown(self):
        if self.widget:
            self.widget.close()
            self.widget.deleteLater()
        self.mc_ErrorDialog.reset_mock()
        super().tearDown()

    def test_Tab_Channels_PushButton_LoadOfflineWindows_DSP(self):
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
        self.widget = FileWidget(
            measurement_file,
            True,  # with_dots
            True,  # subplots
            True,  # subplots_link
            False,  # ignore_value2text_conversions
            False,  # display_cg_name
            "line",  # line_interconnect
            1,  # password
            None,  # hide_missing_channels
            None,  # hide_disabled_channels
        )

        with mock.patch(
            "asammdf.gui.widgets.file.QtWidgets.QFileDialog.getOpenFileName"
        ) as mo_getOpenFileName:
            mo_getOpenFileName.return_value = valid_dsp, None
            QtTest.QTest.mouseClick(
                self.widget.load_channel_list_btn, QtCore.Qt.MouseButton.LeftButton
            )
            # Evaluate
            self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)
            widget_types = sorted(
                map(
                    lambda w: w.widget().__class__.__name__,
                    self.widget.mdi_area.subWindowList(),
                )
            )
            self.assertIn("Plot", widget_types)

    @unittest.skipIf(
        sys.platform in ("darwin", "linux"),
        "Test is failing due to Segmentation Fault on Linux platform.",
    )
    @mock.patch("asammdf.gui.widgets.file.ErrorDialog")
    def test_Tab_Channels_PushButton_LoadOfflineWindows_DSPF(self, mc_file_ErrorDialog):
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
        invalid_json_decode_error_dspf = str(
            pathlib.Path(self.resource, "invalid_JsonDecodeError.dspf")
        )
        invalid_numeric_section_key_error_dspf = str(
            pathlib.Path(self.resource, "invalid_NumericSectionKeyError.dspf")
        )
        invalid_plot_section_key_error_dspf = str(
            pathlib.Path(self.resource, "invalid_PlotSectionKeyError.dspf")
        )
        invalid_tabular_section_key_error_dspf = str(
            pathlib.Path(self.resource, "invalid_TabularSectionKeyError.dspf")
        )

        # Event
        self.widget = FileWidget(
            measurement_file,
            True,  # with_dots
            True,  # subplots
            True,  # subplots_link
            False,  # ignore_value2text_conversions
            False,  # display_cg_name
            "line",  # line_interconnect
            1,  # password
            None,  # hide_missing_channels
            None,  # hide_disabled_channels
        )
        # Case 0
        with self.subTest("test_PushButton_LoadOfflineWindows_DSPF_0"):
            with mock.patch.object(
                self.widget, "load_window", wraps=self.widget.load_window
            ) as mo_load_window, mock.patch(
                "asammdf.gui.widgets.file.QtWidgets.QFileDialog.getOpenFileName"
            ) as mo_getOpenFileName:
                mo_getOpenFileName.return_value = None, None
                QtTest.QTest.mouseClick(
                    self.widget.load_channel_list_btn, QtCore.Qt.MouseButton.LeftButton
                )
                # Evaluate
                mo_load_window.assert_not_called()
                self.assertEqual(len(self.widget.mdi_area.subWindowList()), 0)

        # Case 1
        with self.subTest("test_PushButton_LoadOfflineWindows_DSPF_1"):
            with mock.patch.object(
                self.widget, "load_window", wraps=self.widget.load_window
            ) as mo_load_window, mock.patch(
                "asammdf.gui.widgets.file.QtWidgets.QFileDialog.getOpenFileName"
            ) as mo_getOpenFileName:
                mo_getOpenFileName.return_value = valid_dspf[:-2], None
                QtTest.QTest.mouseClick(
                    self.widget.load_channel_list_btn, QtCore.Qt.MouseButton.LeftButton
                )
                # Evaluate
                mo_load_window.assert_not_called()
                self.assertEqual(len(self.widget.mdi_area.subWindowList()), 0)

        # Case 2
        with self.subTest("test_PushButton_LoadOfflineWindows_DSPF_2"):
            with mock.patch.object(
                self.widget, "load_window", wraps=self.widget.load_window
            ) as mo_load_window, mock.patch(
                "asammdf.gui.widgets.file.QtWidgets.QFileDialog.getOpenFileName"
            ) as mo_getOpenFileName:
                mo_getOpenFileName.return_value = invalid_json_decode_error_dspf, None
                QtTest.QTest.mouseClick(
                    self.widget.load_channel_list_btn, QtCore.Qt.MouseButton.LeftButton
                )
                # Evaluate
                mo_load_window.assert_not_called()
                self.assertEqual(len(self.widget.mdi_area.subWindowList()), 0)
                self.mc_ErrorDialog.assert_called()
                self.mc_ErrorDialog.reset_mock()

        # Case 3
        with self.subTest("test_PushButton_LoadOfflineWindows_DSPF_3"):
            with mock.patch(
                "asammdf.gui.widgets.file.ErrorDialog"
            ) as mc_ErrorDialog, mock.patch(
                "asammdf.gui.widgets.file.QtWidgets.QFileDialog.getOpenFileName"
            ) as mo_getOpenFileName:
                mo_getOpenFileName.return_value = (
                    invalid_plot_section_key_error_dspf,
                    None,
                )
                QtTest.QTest.mouseClick(
                    self.widget.load_channel_list_btn, QtCore.Qt.MouseButton.LeftButton
                )
                # Evaluate
                self.assertEqual(len(self.widget.mdi_area.subWindowList()), 2)
                mc_ErrorDialog.assert_called()
                mc_ErrorDialog.reset_mock()
                widget_types = sorted(
                    map(
                        lambda w: w.widget().__class__.__name__,
                        self.widget.mdi_area.subWindowList(),
                    )
                )
                self.assertNotIn("Plot", widget_types)

        # Case 4
        with self.subTest("test_PushButton_LoadOfflineWindows_DSPF_4"):
            with mock.patch(
                "asammdf.gui.widgets.file.ErrorDialog"
            ) as mc_ErrorDialog, mock.patch(
                "asammdf.gui.widgets.file.QtWidgets.QFileDialog.getOpenFileName"
            ) as mo_getOpenFileName:
                mo_getOpenFileName.return_value = (
                    invalid_numeric_section_key_error_dspf,
                    None,
                )
                QtTest.QTest.mouseClick(
                    self.widget.load_channel_list_btn, QtCore.Qt.MouseButton.LeftButton
                )
                # Evaluate
                self.assertEqual(len(self.widget.mdi_area.subWindowList()), 2)
                mc_ErrorDialog.assert_called()
                mc_ErrorDialog.reset_mock()
                widget_types = sorted(
                    map(
                        lambda w: w.widget().__class__.__name__,
                        self.widget.mdi_area.subWindowList(),
                    )
                )
                self.assertNotIn("Numeric", widget_types)

        # Case 5
        with self.subTest("test_PushButton_LoadOfflineWindows_DSPF_5"):
            with mock.patch(
                "asammdf.gui.widgets.file.ErrorDialog"
            ) as mc_ErrorDialog, mock.patch(
                "asammdf.gui.widgets.file.QtWidgets.QFileDialog.getOpenFileName"
            ) as mo_getOpenFileName:
                mo_getOpenFileName.return_value = (
                    invalid_tabular_section_key_error_dspf,
                    None,
                )
                QtTest.QTest.mouseClick(
                    self.widget.load_channel_list_btn, QtCore.Qt.MouseButton.LeftButton
                )
                # Evaluate
                self.assertEqual(len(self.widget.mdi_area.subWindowList()), 2)
                mc_ErrorDialog.assert_called()
                mc_ErrorDialog.reset_mock()
                widget_types = sorted(
                    map(
                        lambda w: w.widget().__class__.__name__,
                        self.widget.mdi_area.subWindowList(),
                    )
                )
                self.assertNotIn("Tabular", widget_types)

        # Case 6
        with self.subTest("test_PushButton_LoadOfflineWindows_DSPF_6"):
            with mock.patch.object(
                self.widget, "load_window", wraps=self.widget.load_window
            ) as mo_load_window, mock.patch(
                "asammdf.gui.widgets.file.QtWidgets.QFileDialog.getOpenFileName"
            ) as mo_getOpenFileName:
                mo_getOpenFileName.return_value = valid_dspf, None
                QtTest.QTest.mouseClick(
                    self.widget.load_channel_list_btn, QtCore.Qt.MouseButton.LeftButton
                )
                # Evaluate
                mo_load_window.assert_called()
                self.assertEqual(len(self.widget.mdi_area.subWindowList()), 3)
                widget_types = sorted(
                    map(
                        lambda w: w.widget().__class__.__name__,
                        self.widget.mdi_area.subWindowList(),
                    )
                )
                self.assertListEqual(["Numeric", "Plot", "Tabular"], widget_types)

        mc_file_ErrorDialog.assert_not_called()

    def test_Tab_Channels_PushButton_LoadOfflineWindows_LAB(self):
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
        invalid_missing_section_lab = str(
            pathlib.Path(self.resource, "invalid_MissingSection.lab")
        )
        invalid_empty_section_lab = str(
            pathlib.Path(self.resource, "invalid_EmptySection.lab")
        )
        measurement_file = str(pathlib.Path(self.resource, "ASAP2_Demo_V171.mf4"))

        # Event
        self.widget = FileWidget(
            measurement_file,
            True,  # with_dots
            True,  # subplots
            True,  # subplots_link
            False,  # ignore_value2text_conversions
            False,  # display_cg_name
            "line",  # line_interconnect
            1,  # password
            None,  # hide_missing_channels
            None,  # hide_disabled_channels
        )
        # Switch ComboBox to "Internal file structure"
        self.widget.channel_view.setCurrentText("Internal file structure")
        # Case 0:
        with self.subTest("test_PushButton_LoadOfflineWindows_LAB_0"):
            with mock.patch(
                "asammdf.gui.widgets.file.QtWidgets.QFileDialog.getOpenFileName"
            ) as mo_getOpenFileName:
                mo_getOpenFileName.return_value = invalid_empty_section_lab, None
                QtTest.QTest.mouseClick(
                    self.widget.load_channel_list_btn, QtCore.Qt.MouseButton.LeftButton
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
                    if item.checkState(0) == QtCore.Qt.Checked:
                        checked_items.append(item.text(0))
                    iterator += 1
                self.assertEqual(0, len(checked_items))
                self.assertNotRegex(
                    str(self.mc_ErrorDialog.mock_calls),
                    r"local variable .* referenced before assignment",
                )

        # Case 1:
        with self.subTest("test_PushButton_LoadOfflineWindows_LAB_1"):
            with mock.patch(
                "asammdf.gui.widgets.file.QtWidgets.QFileDialog.getOpenFileName"
            ) as mo_getOpenFileName:
                mo_getOpenFileName.return_value = invalid_missing_section_lab, None
                QtTest.QTest.mouseClick(
                    self.widget.load_channel_list_btn, QtCore.Qt.MouseButton.LeftButton
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
                    if item.checkState(0) == QtCore.Qt.Checked:
                        checked_items.append(item.text(0))
                    iterator += 1
                self.assertEqual(0, len(checked_items))
                self.assertNotRegex(
                    str(self.mc_ErrorDialog.mock_calls),
                    r"local variable .* referenced before assignment",
                )

        # Case 2:
        with self.subTest("test_PushButton_LoadOfflineWindows_LAB_2"):
            with mock.patch(
                "asammdf.gui.widgets.file.QtWidgets.QFileDialog.getOpenFileName"
            ) as mo_getOpenFileName, mock.patch(
                "asammdf.gui.widgets.file.QtWidgets.QInputDialog.getItem"
            ) as mo_getItem:
                mo_getOpenFileName.return_value = valid_lab, None
                mo_getItem.return_value = "lab", True
                QtTest.QTest.mouseClick(
                    self.widget.load_channel_list_btn, QtCore.Qt.MouseButton.LeftButton
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
                    if item.checkState(0) == QtCore.Qt.Checked:
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

    def test_Tab_Channels_PushButton_SaveOfflineWindows(self):
        """

        Events:
            - Open 'FileWidget' with valid measurement.
            - Ensure that Channels View is set to "Internal file structure"
            - Press PushButton: "Load offline windows"
                - Simulate that valid "dspf" file was selected
            - Close all Numeric and Tabular windows
            - Add new Plot window
                - Drag and drop signal from channels tree to Plot
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

        # Event
        self.widget = FileWidget(
            measurement_file,
            True,  # with_dots
            True,  # subplots
            True,  # subplots_link
            False,  # ignore_value2text_conversions
            False,  # display_cg_name
            "line",  # line_interconnect
            1,  # password
            None,  # hide_missing_channels
            None,  # hide_disabled_channels
        )
        self.widget.showNormal()
        self.widget.activateWindow()
        # Switch ComboBox to "Internal file structure"
        self.widget.channel_view.setCurrentText("Internal file structure")

        with mock.patch.object(
            self.widget, "load_window", wraps=self.widget.load_window
        ) as mo_load_window, mock.patch(
            "asammdf.gui.widgets.file.QtWidgets.QFileDialog.getOpenFileName"
        ) as mo_getOpenFileName:
            mo_getOpenFileName.return_value = valid_dspf, None
            QtTest.QTest.mouseClick(
                self.widget.load_channel_list_btn, QtCore.Qt.MouseButton.LeftButton
            )
            # Pre-Evaluate
            mo_load_window.assert_called()
            self.assertEqual(len(self.widget.mdi_area.subWindowList()), 3)
            for window in self.widget.mdi_area.subWindowList():
                if window.widget().__class__.__name__ in ("Numeric", "Tabular"):
                    window.close()

        # Drag Part
        channels_tree = self.widget.channels_tree
        mdi_area = self.widget.mdi_area

        iterator = QtWidgets.QTreeWidgetItemIterator(channels_tree)
        while iterator.value():
            item = iterator.value()
            # Expand Parent: Channel group 2 (Engine_1)
            if item.parent() is None and item.text(0) == "Channel group 2 (Engine_1)":
                item.setExpanded(True)
                iterator += 1
                continue
            # Select item: ASAM.M.SCALAR.UBYTE.VTAB_RANGE_NO_DEFAULT_VALUE
            if item.text(0) == "ASAM.M.SCALAR.UBYTE.VTAB_RANGE_NO_DEFAULT_VALUE":
                item.setSelected(True)

                item_rect = channels_tree.visualItemRect(item)
                drag_position = item_rect.center()
                drop_position = mdi_area.viewport().rect().center() - QtCore.QPoint(
                    200, 200
                )

                with mock.patch(
                    "asammdf.gui.widgets.mdi_area.WindowSelectionDialog"
                ) as mc_WindowSelectionDialog:
                    # Setup
                    mc_WindowSelectionDialog.return_value.result.return_value = True
                    mc_WindowSelectionDialog.return_value.disable_new_channels.return_value = (
                        False
                    )
                    mc_WindowSelectionDialog.return_value.selected_type.return_value = (
                        "Plot"
                    )

                    DragAndDrop(
                        source_widget=channels_tree,
                        destination_widget=mdi_area,
                        source_pos=drag_position,
                        destination_pos=drop_position,
                    )
                break
            iterator += 1

        # Press PushButton: "Save offline windows"
        with mock.patch(
            "asammdf.gui.widgets.file.QtWidgets.QFileDialog.getSaveFileName"
        ) as mo_getSaveFileName:
            mo_getSaveFileName.return_value = str(saved_dspf), None
            QtTest.QTest.mouseClick(
                self.widget.save_channel_list_btn, QtCore.Qt.MouseButton.LeftButton
            )
        # Evaluate
        self.assertTrue(saved_dspf.exists())

        # Event
        with mock.patch.object(
            self.widget, "load_window", wraps=self.widget.load_window
        ) as mo_load_window, mock.patch(
            "asammdf.gui.widgets.file.QtWidgets.QFileDialog.getOpenFileName"
        ) as mo_getOpenFileName:
            mo_getOpenFileName.return_value = saved_dspf, None
            QtTest.QTest.mouseClick(
                self.widget.load_channel_list_btn, QtCore.Qt.MouseButton.LeftButton
            )
            # Pre-Evaluate
            mo_load_window.assert_called()
            self.assertEqual(len(self.widget.mdi_area.subWindowList()), 2)
            widget_types = set(
                map(
                    lambda w: w.widget().__class__.__name__,
                    self.widget.mdi_area.subWindowList(),
                )
            )
            self.assertSetEqual({"Plot"}, widget_types)

    def test_Tab_Channels_PushButton_SelectAll(self):
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
        self.widget = FileWidget(
            measurement_file,
            True,  # with_dots
            True,  # subplots
            True,  # subplots_link
            False,  # ignore_value2text_conversions
            False,  # display_cg_name
            "line",  # line_interconnect
            1,  # password
            None,  # hide_missing_channels
            None,  # hide_disabled_channels
        )
        self.widget.showNormal()

        # Switch ComboBox to "Natural sort"
        self.widget.channel_view.setCurrentText("Natural Sort")
        # Press PushButton: "Select all the channels"
        QtTest.QTest.mouseClick(
            self.widget.select_all_btn, QtCore.Qt.MouseButton.LeftButton
        )

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
            item.setCheckState(0, QtCore.Qt.Unchecked)
            self.assertFalse(item.checkState(0))
            iterator += 1

        # Switch ComboBox to "Internal file structure"
        self.widget.channel_view.setCurrentText("Internal file structure")
        # Press PushButton: "Select all the channels"
        QtTest.QTest.mouseClick(
            self.widget.select_all_btn, QtCore.Qt.MouseButton.LeftButton
        )

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

    def test_Tab_Channels_PushButton_ClearAll(self):
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
        self.widget = FileWidget(
            measurement_file,
            True,  # with_dots
            True,  # subplots
            True,  # subplots_link
            False,  # ignore_value2text_conversions
            False,  # display_cg_name
            "line",  # line_interconnect
            1,  # password
            None,  # hide_missing_channels
            None,  # hide_disabled_channels
        )
        self.widget.showNormal()

        # Switch ComboBox to "Natural sort"
        self.widget.channel_view.setCurrentText("Natural Sort")
        # Select all
        iterator = QtWidgets.QTreeWidgetItemIterator(self.widget.channels_tree)
        while iterator.value():
            item = iterator.value()
            item.setCheckState(0, QtCore.Qt.Checked)
            self.assertTrue(item.checkState(0))
            iterator += 1
        # Press PushButton: "Clear all selected channels"
        QtTest.QTest.mouseClick(
            self.widget.clear_channels_btn, QtCore.Qt.MouseButton.LeftButton
        )

        # Evaluate
        iterator = QtWidgets.QTreeWidgetItemIterator(self.widget.channels_tree)
        while iterator.value():
            item = iterator.value()
            self.assertFalse(item.checkState(0))
            iterator += 1

        # Switch ComboBox to "Internal file structure"
        self.widget.channel_view.setCurrentText("Internal file structure")
        while iterator.value():
            item = iterator.value()
            item.setCheckState(0, QtCore.Qt.Checked)
            self.assertTrue(item.checkState(0))
            iterator += 1
        # Press PushButton: "Clear all selected channels"
        QtTest.QTest.mouseClick(
            self.widget.clear_channels_btn, QtCore.Qt.MouseButton.LeftButton
        )

        # Evaluate
        iterator = QtWidgets.QTreeWidgetItemIterator(self.widget.channels_tree)
        while iterator.value():
            item = iterator.value()
            self.assertFalse(item.checkState(0))
            iterator += 1

        # Switch ComboBox to "Selected channels only"
        self.widget.channel_view.setCurrentText("Selected channels only")

        # Evaluate
        iterator = QtWidgets.QTreeWidgetItemIterator(self.widget.channels_tree)
        while iterator.value():
            item = iterator.value()
            self.assertFalse(item.checkState(0))
            iterator += 1

    def test_Tab_Channels_PushButton_Search(self):
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
        self.widget = FileWidget(
            measurement_file,
            True,  # with_dots
            True,  # subplots
            True,  # subplots_link
            False,  # ignore_value2text_conversions
            False,  # display_cg_name
            "line",  # line_interconnect
            1,  # password
            None,  # hide_missing_channels
            None,  # hide_disabled_channels
        )
        self.widget.showNormal()
        # Case 0:
        with self.subTest("test_Tab_Channels_PushButton_Search_0"):
            with mock.patch(
                "asammdf.gui.widgets.file.AdvancedSearch"
            ) as mc_AdvancedSearch:
                mc_AdvancedSearch.return_value.result = {}
                mc_AdvancedSearch.return_value.pattern_window = False
                mc_AdvancedSearch.return_value.add_window_request = False

                # - Press PushButton: "Search and select channels"
                QtTest.QTest.mouseClick(
                    self.widget.advanced_search_btn, QtCore.Qt.LeftButton
                )
                # Evaluate
                iterator = QtWidgets.QTreeWidgetItemIterator(self.widget.channels_tree)
                while iterator.value():
                    item = iterator.value()
                    self.assertFalse(item.checkState(0))
                    iterator += 1

        # Case 1:
        with self.subTest("test_Tab_Channels_PushButton_Search_1"):
            with mock.patch(
                "asammdf.gui.widgets.file.AdvancedSearch"
            ) as mc_AdvancedSearch:
                mc_AdvancedSearch.return_value.result = {
                    (4, 3): "ASAM.M.SCALAR.FLOAT64.IDENTICAL",
                    (2, 10): "ASAM.M.SCALAR.FLOAT32.IDENTICAL",
                }
                mc_AdvancedSearch.return_value.pattern_window = False
                mc_AdvancedSearch.return_value.add_window_request = False

                # - Press PushButton: "Search and select channels"
                QtTest.QTest.mouseClick(
                    self.widget.advanced_search_btn, QtCore.Qt.LeftButton
                )
                # Evaluate
                checked_channels = 0
                iterator = QtWidgets.QTreeWidgetItemIterator(self.widget.channels_tree)
                while iterator.value():
                    item = iterator.value()
                    if item.checkState(0) == QtCore.Qt.Checked:
                        checked_channels += 1
                    iterator += 1
                self.assertEqual(2, checked_channels)
                self.assertEqual(len(self.widget.mdi_area.subWindowList()), 0)

        # Case 2:
        with self.subTest("test_Tab_Channels_PushButton_Search_2"):
            # - Clear 'channel_tree' selection
            iterator = QtWidgets.QTreeWidgetItemIterator(self.widget.channels_tree)
            while iterator.value():
                item = iterator.value()
                item.setCheckState(0, QtCore.Qt.Unchecked)
                iterator += 1
            with mock.patch(
                "asammdf.gui.widgets.file.AdvancedSearch"
            ) as mc_AdvancedSearch, mock.patch(
                "asammdf.gui.widgets.file.WindowSelectionDialog"
            ) as mc_WindowSelectionDialog:
                mc_AdvancedSearch.return_value.result = {
                    (4, 3): "ASAM.M.SCALAR.FLOAT64.IDENTICAL",
                    (2, 10): "ASAM.M.SCALAR.FLOAT32.IDENTICAL",
                }
                mc_AdvancedSearch.return_value.pattern_window = False
                mc_AdvancedSearch.return_value.add_window_request = True
                mc_WindowSelectionDialog.return_value.result.return_value = True
                mc_WindowSelectionDialog.return_value.selected_type.return_value = (
                    "New plot window"
                )
                mc_WindowSelectionDialog.return_value.disable_new_channels.return_value = (
                    False
                )

                # - Press PushButton: "Search and select channels"
                QtTest.QTest.mouseClick(
                    self.widget.advanced_search_btn, QtCore.Qt.LeftButton
                )
                # Evaluate
                checked_channels = 0
                iterator = QtWidgets.QTreeWidgetItemIterator(self.widget.channels_tree)
                while iterator.value():
                    item = iterator.value()
                    if item.checkState(0) == QtCore.Qt.Checked:
                        checked_channels += 1
                    iterator += 1
                self.assertEqual(2, checked_channels)
                self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)
                widget_types = sorted(
                    map(
                        lambda w: w.widget().__class__.__name__,
                        self.widget.mdi_area.subWindowList(),
                    )
                )
                self.assertIn("Plot", widget_types)

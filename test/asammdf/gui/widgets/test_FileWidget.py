import pathlib
from test.asammdf.gui import QtCore, QtTest, QtWidgets
from test.asammdf.gui.test_base import TestBase
from unittest import mock

from PySide6 import QtCore, QtWidgets

from asammdf.gui.widgets.file import FileWidget


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
        super().tearDown()

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

        with (
            mock.patch(
                "src.asammdf.gui.widgets.file.QtWidgets.QFileDialog.getOpenFileName"
            ) as mo_getOpenFileName,
        ):
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

    def test_PushButton_LoadOfflineWindows_DSPF(self):
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
            with (
                mock.patch.object(
                    self.widget, "load_window", wraps=self.widget.load_window
                ) as mo_load_window,
                mock.patch(
                    "src.asammdf.gui.widgets.file.QtWidgets.QFileDialog.getOpenFileName"
                ) as mo_getOpenFileName,
            ):
                mo_getOpenFileName.return_value = None, None
                QtTest.QTest.mouseClick(
                    self.widget.load_channel_list_btn, QtCore.Qt.MouseButton.LeftButton
                )
                # Evaluate
                mo_load_window.assert_not_called()
                self.assertEqual(len(self.widget.mdi_area.subWindowList()), 0)

        # Case 1
        with self.subTest("test_PushButton_LoadOfflineWindows_DSPF_1"):
            with (
                mock.patch.object(
                    self.widget, "load_window", wraps=self.widget.load_window
                ) as mo_load_window,
                mock.patch(
                    "src.asammdf.gui.widgets.file.QtWidgets.QFileDialog.getOpenFileName"
                ) as mo_getOpenFileName,
            ):
                mo_getOpenFileName.return_value = valid_dspf[:-2], None
                QtTest.QTest.mouseClick(
                    self.widget.load_channel_list_btn, QtCore.Qt.MouseButton.LeftButton
                )
                # Evaluate
                mo_load_window.assert_not_called()
                self.assertEqual(len(self.widget.mdi_area.subWindowList()), 0)

        # Case 2
        with self.subTest("test_PushButton_LoadOfflineWindows_DSPF_2"):
            with (
                mock.patch.object(
                    self.widget, "load_window", wraps=self.widget.load_window
                ) as mo_load_window,
                mock.patch(
                    "src.asammdf.gui.widgets.file.QtWidgets.QFileDialog.getOpenFileName"
                ) as mo_getOpenFileName,
            ):
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
            with (
                # Magic. Trust me.
                mock.patch("asammdf.gui.widgets.file.ErrorDialog") as mc_ErrorDialog,
                mock.patch(
                    "src.asammdf.gui.widgets.file.QtWidgets.QFileDialog.getOpenFileName"
                ) as mo_getOpenFileName,
            ):
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
            with (
                # Magic. Trust me.
                mock.patch("asammdf.gui.widgets.file.ErrorDialog") as mc_ErrorDialog,
                mock.patch(
                    "src.asammdf.gui.widgets.file.QtWidgets.QFileDialog.getOpenFileName"
                ) as mo_getOpenFileName,
            ):
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
            with (
                # Magic. Trust me.
                mock.patch("asammdf.gui.widgets.file.ErrorDialog") as mc_ErrorDialog,
                mock.patch(
                    "src.asammdf.gui.widgets.file.QtWidgets.QFileDialog.getOpenFileName"
                ) as mo_getOpenFileName,
            ):
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
            with (
                mock.patch.object(
                    self.widget, "load_window", wraps=self.widget.load_window
                ) as mo_load_window,
                mock.patch(
                    "src.asammdf.gui.widgets.file.QtWidgets.QFileDialog.getOpenFileName"
                ) as mo_getOpenFileName,
            ):
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

    def test_PushButton_LoadOfflineWindows_LAB(self):
        """
        Events:
            - Open 'FileWidget' with valid measurement.
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
        # Case 0:
        with self.subTest("test_PushButton_LoadOfflineWindows_LAB_0"):
            with (
                mock.patch(
                    "src.asammdf.gui.widgets.file.QtWidgets.QFileDialog.getOpenFileName"
                ) as mo_getOpenFileName,
            ):
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
            with (
                mock.patch(
                    "src.asammdf.gui.widgets.file.QtWidgets.QFileDialog.getOpenFileName"
                ) as mo_getOpenFileName,
            ):
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
            with (
                mock.patch(
                    "src.asammdf.gui.widgets.file.QtWidgets.QFileDialog.getOpenFileName"
                ) as mo_getOpenFileName,
                mock.patch(
                    "asammdf.gui.widgets.file.QtWidgets.QInputDialog.getItem"
                ) as mo_getItem,
            ):
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

    def test_PushButton_SaveOfflineWindows(self):
        """
        Events:
            - Open 'FileWidget' with valid measurement.
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
        self.widget.showMaximized()
        self.widget.activateWindow()

        with (
            mock.patch.object(
                self.widget, "load_window", wraps=self.widget.load_window
            ) as mo_load_window,
            mock.patch(
                "src.asammdf.gui.widgets.file.QtWidgets.QFileDialog.getOpenFileName"
            ) as mo_getOpenFileName,
        ):
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

                # item_rect = channels_tree.visualItemRect(item)
                # drag_position = item_rect.center()
                drop_position = mdi_area.viewport().rect().center() - QtCore.QPoint(200, 200)

                # Press on item
                # Don't know how to trigger startDrag for now.
                # QtTest.QTest.mousePress(
                #     channels_tree,
                #     QtCore.Qt.LeftButton,
                #     QtCore.Qt.NoModifier,
                #     item_center
                # )
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

                    channels_tree.startDrag(QtCore.Qt.MoveAction)
                    # Move item
                    QtTest.QTest.mouseMove(mdi_area, drop_position)
                    # Release item
                    QtTest.QTest.mouseRelease(
                        mdi_area,
                        QtCore.Qt.LeftButton,
                        QtCore.Qt.NoModifier,
                        drop_position,
                    )
                break
            iterator += 1

        QtTest.QTest.qWait(10)
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
        with (
            mock.patch.object(
                self.widget, "load_window", wraps=self.widget.load_window
            ) as mo_load_window,
            mock.patch(
                "src.asammdf.gui.widgets.file.QtWidgets.QFileDialog.getOpenFileName"
            ) as mo_getOpenFileName,
        ):
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

import inspect
import json
import pathlib
from unittest import mock

from PySide6 import QtTest, QtCore

from asammdf.gui.dialogs.functions_manager import FunctionsManagerDialog
from test.asammdf.gui.resources.functions import (
    Function1,
    maximum,
    gray2dec,
    rpm_to_rad_per_second,
    Function2,
    UnresolvedVariable,
    WrongDefinition,
)
from test.asammdf.gui.test_base import TestBase


class TestPushButtons(TestBase):
    def setUp(self) -> None:
        super().setUp()
        self.fm = FunctionsManagerDialog({})

    def test_PushButton_Add(self):
        """
        Test Scope:
            - Ensure that Functions can be added.
        Precondition:
            - Open Function Manager Widget
        Events:
            - Press PushButton "Add"
            - Press PushButton "Add"
        Evaluate:
            - Ensure that two items will be added in functions list
        """
        # Events:
        QtTest.QTest.mouseClick(self.fm.widget.add_btn, QtCore.Qt.MouseButton.LeftButton)
        QtTest.QTest.mouseClick(self.fm.widget.add_btn, QtCore.Qt.MouseButton.LeftButton)

        # Evaluate
        self.assertEqual(2, self.fm.widget.functions_list.count())
        for index in range(3, 1):
            self.assertEqual(f"Function{index}", self.fm.widget.functions_list.item(index - 1))

    def test_PushButton_DeleteAll(self):
        """
        Test Scope:
            - Ensure that all Functions are deleted via PushButton DeleteAll.
        Preconditions:
            - Open Function Manager Widget
        Events:
            - Press PushButton "Add"
            - Press PushButton "Add"
            - Press PushButton "DeleteAll"
        Evaluate:
            - Evaluate that items are deleted from functions list.
        """
        # Events:
        QtTest.QTest.mouseClick(self.fm.widget.add_btn, QtCore.Qt.MouseButton.LeftButton)
        QtTest.QTest.mouseClick(self.fm.widget.add_btn, QtCore.Qt.MouseButton.LeftButton)

        self.assertEqual(2, self.fm.widget.functions_list.count())
        for index in range(3, 1):
            self.assertEqual(f"Function{index}", self.fm.widget.functions_list.item(index - 1))

        QtTest.QTest.mouseClick(self.fm.widget.erase_btn, QtCore.Qt.MouseButton.LeftButton)

        # Evaluate
        self.assertEqual(0, self.fm.widget.functions_list.count())

    def test_PushButton_LoadDefinitions_0(self):
        """
        Test Scope:
            - Ensure that Functions from '.def' file are loaded in function list
        Preconditions:
            - Open Function Manager Widget
        Events:
            - Press PushButton "Load Definitions"
                - Simulate that valid path is provided.
        Evaluate:
            - Evaluate that files are loaded into function list.
        """
        definition_file = pathlib.Path(self.resource, "functions_definitions.def")
        with mock.patch(
            "asammdf.gui.widgets.functions_manager.QtWidgets.QFileDialog.getOpenFileName"
        ) as mc_getOpenFileName:
            mc_getOpenFileName.return_value = definition_file, None
            QtTest.QTest.mouseClick(self.fm.widget.import_btn, QtCore.Qt.MouseButton.LeftButton)

        # Evaluate
        self.assertEqual(4, self.fm.widget.functions_list.count())
        self.assertEqual("Function1", self.fm.widget.functions_list.item(0).text())
        self.assertEqual("gray2dec", self.fm.widget.functions_list.item(1).text())
        self.assertEqual("maximum", self.fm.widget.functions_list.item(2).text())
        self.assertEqual("rpm_to_rad_per_second", self.fm.widget.functions_list.item(3).text())

    def test_PushButton_LoadDefinitions_1(self):
        """
        Test Scope:
            - Ensure that Functions from nonexistent '.def' file is handled.
        Preconditions:
            - Open Function Manager Widget
        Events:
            - Press PushButton "Load Definitions"
                - Simulate that valid path is provided but file does not exist.
        Evaluate:
            - Evaluate that functions from file are loaded into function list.
        """
        definition_file = pathlib.Path(self.test_workspace, "nonexistent.def")
        with mock.patch(
            "asammdf.gui.widgets.functions_manager.QtWidgets.QFileDialog.getOpenFileName"
        ) as mc_getOpenFileName:
            mc_getOpenFileName.return_value = definition_file, None
            QtTest.QTest.mouseClick(self.fm.widget.import_btn, QtCore.Qt.MouseButton.LeftButton)

        # Evaluate
        self.assertEqual(0, self.fm.widget.functions_list.count())

    def test_PushButton_LoadDefinitions_2(self):
        """
        Test Scope:
            - Ensure that case where import dialog is closed, is handled.
        Preconditions:
            - Open Function Manager Widget
        Events:
            - Press PushButton "Load Definitions"
                - Simulate that no valid path is provided.
        Evaluate:
            - Evaluate that no functions is loaded to function list.
        """
        with mock.patch(
            "asammdf.gui.widgets.functions_manager.QtWidgets.QFileDialog.getOpenFileName"
        ) as mc_getOpenFileName:
            mc_getOpenFileName.return_value = None, None
            QtTest.QTest.mouseClick(self.fm.widget.import_btn, QtCore.Qt.MouseButton.LeftButton)

        # Evaluate
        self.assertEqual(0, self.fm.widget.functions_list.count())

    def test_PushButton_LoadDefinitions_3(self):
        """
        Test Scope:
            - Ensure that Functions from '.def' file are loaded in function list
        Preconditions:
            - Open Function Manager Widget
        Events:
            - Press PushButton "Load Definitions"
                - Simulate that valid path is provided but file does not have correct structure.
        Evaluate:
            - Evaluate that no functions is loaded to function list.
        """
        definition_file = pathlib.Path(self.test_workspace, "invalid.def")
        with open(definition_file, "w+") as fpw:
            fpw.write(self.id())

        with mock.patch(
            "asammdf.gui.widgets.functions_manager.QtWidgets.QFileDialog.getOpenFileName"
        ) as mc_getOpenFileName:
            mc_getOpenFileName.return_value = definition_file, None
            QtTest.QTest.mouseClick(self.fm.widget.import_btn, QtCore.Qt.MouseButton.LeftButton)

        # Evaluate
        self.assertEqual(0, self.fm.widget.functions_list.count())
        self.mc_ErrorDialog.assert_called()

    def test_PushButton_LoadDefinitions_4(self):
        """
        Test Scope:
            - Ensure that Functions from file with extension different from '.def' are not loaded in function list
        Preconditions:
            - Open Function Manager Widget
        Events:
            - Press PushButton "Load Definitions"
                - Simulate that valid path is provided but file does not have correct structure.
        Evaluate:
            - Evaluate that no functions is loaded to function list.
        """
        definition_file = pathlib.Path(self.test_workspace, "wrong_extension.deff")
        with open(definition_file, "w+") as fpw:
            json.dump({"Function1": "def Function1(t=0):\n    return 0"}, fpw, sort_keys=True, indent=2)

        with mock.patch(
            "asammdf.gui.widgets.functions_manager.QtWidgets.QFileDialog.getOpenFileName"
        ) as mc_getOpenFileName:
            mc_getOpenFileName.return_value = definition_file, None
            QtTest.QTest.mouseClick(self.fm.widget.import_btn, QtCore.Qt.MouseButton.LeftButton)

        # Evaluate
        self.assertEqual(0, self.fm.widget.functions_list.count())

    def test_PushButton_SaveDefinitions(self):
        """
        Test Scope:
            - Ensure that Functions from '.def' file are loaded in function list
        Preconditions:
            - Open Function Manager Widget
        Events:
            - Press PushButton Add
            - Press PushButton Save Definitions
        Evaluate:
            - Evaluate that dialog is open and file can be saved.
            - Evaluate that new function is present in saved file.
        """
        # Event
        QtTest.QTest.mouseClick(self.fm.widget.add_btn, QtCore.Qt.MouseButton.LeftButton)
        saved_file = pathlib.Path(self.test_workspace, "saved_file.def")
        with mock.patch(
            "asammdf.gui.widgets.functions_manager.QtWidgets.QFileDialog.getSaveFileName"
        ) as mc_getSaveFileName:
            mc_getSaveFileName.return_value = str(saved_file), None
            QtTest.QTest.mouseClick(self.fm.widget.export_btn, QtCore.Qt.MouseButton.LeftButton)

        # Evaluate
        self.assertTrue(saved_file.exists())
        with open(saved_file, "r") as fpr:
            content = json.load(fpr)
            self.assertDictEqual(content, {"Function1": "def Function1(t=0):\n    return 0"})

    def test_PushButton_CheckSyntax_0(self):
        """
        Test Scope:
            - Ensure that valid Python Syntax is detected.
        Events:
            - Press PushButton Add
            - Edit Function1 content. Add valid python syntax
            - Press PushButton Check Syntax
        Evaluate:
            - Evaluate that python function is evaluated as valid.
        """
        # Event
        QtTest.QTest.mouseClick(self.fm.widget.add_btn, QtCore.Qt.MouseButton.LeftButton)

        for f in (
            Function1,
            gray2dec,
            maximum,
            rpm_to_rad_per_second,
        ):
            with self.subTest(f"{self.id}_{f.__name__}"):
                self.mc_ErrorDialog.reset_mock()

                source = inspect.getsource(f)
                self.fm.widget.function_definition.clear()
                self.fm.widget.function_definition.setPlainText(source)

                with mock.patch("asammdf.gui.utils.QtWidgets.QMessageBox.information") as mc_information:
                    QtTest.QTest.mouseClick(self.fm.widget.check_syntax_btn, QtCore.Qt.MouseButton.LeftButton)

                    # Evaluate
                    self.mc_ErrorDialog.assert_not_called()
                    mc_information.assert_called()

    def test_PushButton_CheckSyntax_1(self):
        """
        Test Scope:
            - Ensure that invalid Python Syntax is detected.
        Events:
            - Press PushButton Add
            - Edit Function1 content. Add valid python syntax
            - Press PushButton Check Syntax
        Evaluate:
            - Evaluate that python function is evaluated as valid.
        """
        # Event
        QtTest.QTest.mouseClick(self.fm.widget.add_btn, QtCore.Qt.MouseButton.LeftButton)

        for f in (
            Function2,
            UnresolvedVariable,
            WrongDefinition,
        ):
            with self.subTest(f"{self.id}_{f.__name__}"):
                self.mc_ErrorDialog.reset_mock()

                source = inspect.getsource(f)
                self.fm.widget.function_definition.clear()
                self.fm.widget.function_definition.setPlainText(source)

                with mock.patch("asammdf.gui.utils.QtWidgets.QMessageBox.information") as mc_information:
                    QtTest.QTest.mouseClick(self.fm.widget.check_syntax_btn, QtCore.Qt.MouseButton.LeftButton)

                    # Evaluate
                    mc_information.assert_not_called()
                    self.mc_ErrorDialog.assert_called()

    def test_PushButton_StoreFunctionChanges(self):
        pass

import copy
import inspect
import json
import pathlib
from unittest import mock

from PySide6 import QtCore, QtTest

from asammdf.gui.dialogs.functions_manager import FunctionsManagerDialog
from test.asammdf.gui.resources.functions import (
    Function2,
    gray2dec,
    maximum,
    rpm_to_rad_per_second,
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
            json.dump(
                {"Function1": "def Function1(t=0):\n    return 0"},
                fpw,
                sort_keys=True,
                indent=2,
            )

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
        saved_file = pathlib.Path(self.test_workspace, f"{self.id()}.def")
        with mock.patch(
            "asammdf.gui.widgets.functions_manager.QtWidgets.QFileDialog.getSaveFileName"
        ) as mc_getSaveFileName:
            mc_getSaveFileName.return_value = str(saved_file), None
            QtTest.QTest.mouseClick(self.fm.widget.export_btn, QtCore.Qt.MouseButton.LeftButton)

        # Evaluate
        self.assertTrue(saved_file.exists())
        with open(saved_file) as fpr:
            content = json.load(fpr)
            self.assertDictEqual(
                content, {"Function1": "def Function1(t=0):\n    return 0", "__global_variables__": ""}
            )

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
            gray2dec,
            maximum,
            rpm_to_rad_per_second,
        ):
            with self.subTest(f"{self.id}_{f.__name__}"):
                self.mc_ErrorDialog.reset_mock()

                source = inspect.getsource(f)
                self.fm.widget.function_definition.clear()
                self.fm.widget.function_definition.setPlainText(source)

                with mock.patch("asammdf.gui.utils.MessageBox.information") as mc_information:
                    QtTest.QTest.mouseClick(
                        self.fm.widget.check_syntax_btn,
                        QtCore.Qt.MouseButton.LeftButton,
                    )

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

                with mock.patch("asammdf.gui.utils.MessageBox.information") as mc_information:
                    QtTest.QTest.mouseClick(
                        self.fm.widget.check_syntax_btn,
                        QtCore.Qt.MouseButton.LeftButton,
                    )

                    # Evaluate
                    mc_information.assert_not_called()
                    self.mc_ErrorDialog.assert_called()

    def test_PushButton_StoreFunctionChanges_0(self):
        """
        Test Scope:
            - Ensure that function changes are saved.
        Events:
            - Press PushButton Add
            - Edit Function content
            - Press PushButton Store Function Changes
            - Press PushButton Save Definitions
        Evaluate:
            - Evaluate that name of the Function was changed.
            - Evaluate that content of the function was changed.
        """
        # Events:
        QtTest.QTest.mouseClick(self.fm.widget.add_btn, QtCore.Qt.MouseButton.LeftButton)
        self.mouseClick_WidgetItem(self.fm.widget.functions_list.item(0))

        source = inspect.getsource(maximum)
        self.fm.widget.function_definition.clear()
        self.fm.widget.function_definition.setPlainText(source)

        QtTest.QTest.mouseClick(self.fm.widget.store_btn, QtCore.Qt.MouseButton.LeftButton)

        saved_file = pathlib.Path(self.test_workspace, f"{self.id()}.def")
        with mock.patch(
            "asammdf.gui.widgets.functions_manager.QtWidgets.QFileDialog.getSaveFileName"
        ) as mc_getSaveFileName:
            mc_getSaveFileName.return_value = str(saved_file), None
            QtTest.QTest.mouseClick(self.fm.widget.export_btn, QtCore.Qt.MouseButton.LeftButton)

        # Evaluate
        self.assertTrue(saved_file.exists())
        with open(saved_file) as fpr:
            content = json.load(fpr)
            self.assertIn(maximum.__name__, content)
            self.assertIn(content["maximum"], source)

    def test_PushButton_StoreFunctionChanges_1(self):
        """
        Test Scope:
            - Ensure that function changes are saved.
        Events:
            - Press PushButton Add
            - Press PushButton Add
            - Edit 2nd Function and rename as 1st Function
            - Press PushButton Store Function Changes
        Evaluate:
            - Evaluate that overwriting is not possible
        """
        # Events:
        QtTest.QTest.mouseClick(self.fm.widget.add_btn, QtCore.Qt.MouseButton.LeftButton)
        QtTest.QTest.mouseClick(self.fm.widget.add_btn, QtCore.Qt.MouseButton.LeftButton)

        self.mouseClick_WidgetItem(self.fm.widget.functions_list.item(0))
        function1 = self.fm.widget.function_definition.toPlainText()

        self.mouseClick_WidgetItem(self.fm.widget.functions_list.item(1))
        self.fm.widget.function_definition.clear()
        self.fm.widget.function_definition.setPlainText(function1)

        with mock.patch("asammdf.gui.widgets.functions_manager.MessageBox.information") as mc_information:
            QtTest.QTest.mouseClick(self.fm.widget.store_btn, QtCore.Qt.MouseButton.LeftButton)

            mc_information.assert_called()

    def test_PushButton_Apply(self):
        """
        Test Scope:
            - Ensure that definitions are saved when apply button is pressed.
        Events:
            - Press PushButton Add
            - Press PushButton Add
            - Update content of current function
            - Press PushButton Store
            - Press PushButton 'Apply'
        Evaluate:
            - Evaluate that definitions attribute is the same.
        """
        # Events
        QtTest.QTest.mouseClick(self.fm.widget.add_btn, QtCore.Qt.MouseButton.LeftButton)
        QtTest.QTest.mouseClick(self.fm.widget.add_btn, QtCore.Qt.MouseButton.LeftButton)
        definitions = copy.deepcopy(self.fm.widget.definitions)

        self.fm.widget.function_definition.clear()
        source = inspect.getsource(maximum)
        self.fm.widget.function_definition.setPlainText(source)
        QtTest.QTest.mouseClick(self.fm.widget.store_btn, QtCore.Qt.MouseButton.LeftButton)
        QtTest.QTest.mouseClick(self.fm.apply_btn, QtCore.Qt.MouseButton.LeftButton)

        self.assertEqual("apply", self.fm.pressed_button)
        self.assertEqual(2, len(self.fm.modified_definitions))

    def test_PushButton_Cancel(self):
        """
        Test Scope:
            - Ensure that definitions are not saved when cancel button is pressed.
        Events:
            - Press PushButton Add
            - Press PushButton Add
            - Update content of current function
            - Press PushButton Store
            - Press PushButton Cancel
        Evaluate:
            - Evaluate that definitions attribute is the same.
        """
        # Events
        QtTest.QTest.mouseClick(self.fm.widget.add_btn, QtCore.Qt.MouseButton.LeftButton)
        QtTest.QTest.mouseClick(self.fm.widget.add_btn, QtCore.Qt.MouseButton.LeftButton)

        self.fm.widget.function_definition.clear()
        source = inspect.getsource(maximum)
        self.fm.widget.function_definition.setPlainText(source)
        QtTest.QTest.mouseClick(self.fm.widget.store_btn, QtCore.Qt.MouseButton.LeftButton)
        QtTest.QTest.mouseClick(self.fm.cancel_btn, QtCore.Qt.MouseButton.LeftButton)

        self.assertEqual("cancel", self.fm.pressed_button)
        self.assertDictEqual({}, self.fm.modified_definitions)


class TestTreeWidget(TestBase):
    def setUp(self) -> None:
        super().setUp()
        self.fm = FunctionsManagerDialog({"Function10": "def Function10(t=0):\n    return 0"})

    def test_KeyPress_Delete(self):
        """
        Test Scope:
            - Ensure that functions can be deleted from function list.
        Events:
            - Press PushButton Add
            - Press PushButton Add
            - Press PushButton Add
            - Select "Function2"
            - Press Key Delete
        Evaluate:
            - Evaluate that "Function2" is no longer part of function list.
        """
        # Events:
        for _ in range(3):
            QtTest.QTest.mouseClick(self.fm.widget.add_btn, QtCore.Qt.MouseButton.LeftButton)
        self.assertEqual(4, self.fm.widget.functions_list.count())

        self.mouseClick_WidgetItem(self.fm.widget.functions_list.item(2))
        QtTest.QTest.keyClick(self.fm.widget.functions_list, QtCore.Qt.Key.Key_Delete)

        # Evaluate
        self.assertEqual(3, self.fm.widget.functions_list.count())
        self.assertNotEqual("Function2", self.fm.widget.functions_list.item(0))
        self.assertNotEqual("Function2", self.fm.widget.functions_list.item(1))

    def test_FunctionDefinition_ContentUpdate(self):
        """
        Test Scope:
            - Ensure that content is updated in Function Definition when Function is changed in FunctionList
        Events:
            - Press PushButton Add
            - Select Function1
            - Select Function10
        Evaluate:
            - Evaluate that content of function definition is changed.
        """
        # Event
        QtTest.QTest.mouseClick(self.fm.widget.add_btn, QtCore.Qt.MouseButton.LeftButton)
        self.assertEqual(2, self.fm.widget.functions_list.count())

        self.mouseClick_WidgetItem(self.fm.widget.functions_list.item(0))
        function1 = self.fm.widget.function_definition.toPlainText()

        self.mouseClick_WidgetItem(self.fm.widget.functions_list.item(1))
        function2 = self.fm.widget.function_definition.toPlainText()

        # Evaluate
        self.assertNotEqual(function1, function2)

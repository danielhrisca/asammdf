#!/usr/bin/env python
import json
from os import path
from random import randint
from unittest import mock

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from asammdf.gui.widgets.functions_manager import FunctionsManager
from test.asammdf.gui.test_base import TestBase


class TestContextMenu(TestBase):
    """
    tests for function_manager widget from asammdf.gui.widgets
    """

    def setUp(self) -> None:
        """
        Events:
            - Open functions manager widget

        Evaluate
            - Evaluate that functions' list is empty
        """
        super().setUp()
        self.widget = FunctionsManager(definitions={})
        self.processEvents()

        # Evaluate
        self.assertEqual(self.widget.functions_list.count(), 0)

    def test_add_btn(self):
        """
        Test scope:
            Check if by pressing 'Add button', new item will be added in a functions' list.

        Events:
            - Press 'Add' button

        Evaluate:
            - Evaluate that one item was added to the functions' list
        """
        # Items at start
        initial_items_count = self.widget.functions_list.count()

        # Event
        QTest.mouseClick(self.widget.add_btn, Qt.MouseButton.LeftButton)
        self.processEvents()

        # Evaluate
        self.assertEqual(self.widget.functions_list.count(), initial_items_count + 1)

    def test_erase_btn(self):
        """
        Test scope:
            Check if by pressing 'Delete all' button, all items will be removed from functions' list.

        Events:
            - Add few items in the functions' list
            - Press 'Delete all' button

        Evaluate:
            - Evaluate that all items are removed from functions' list
        """
        # Setup
        items_count = randint(1, 10)
        for _ in range(items_count):
            self.widget.add_definition()
        self.processEvents()
        # Quick evaluation
        self.assertEqual(self.widget.functions_list.count(), items_count)

        # Event
        QTest.mouseClick(self.widget.erase_btn, Qt.MouseButton.LeftButton)
        self.processEvents()

        # Evaluate
        self.assertEqual(self.widget.functions_list.count(), 0)

    def test_import_btn(self):
        """
        Test scope:
            Check if by pressing 'Load definitions' button,
            all functions definitions from "*.def" file will be added.

        Events:
            - Open invalid file (not *.def file)
            - Open valid file (*.def file)

        Evaluate:
            - For invalid file:
                > Evaluate that open file method was called
                > Evaluate that there are not function definitions loaded
            - For valid file:
                > Evaluate that open file method was called
                > Evaluate that all definitions from file were loaded
                > Evaluate that names of functions from .def file is in functions list
        """
        # Setup
        # File with functions definitions
        ok_file_path = path.join(self.resource, "functions_definitions.def")
        not_ok_file_path = path.join(self.resource, "saved.dspf")
        with open(ok_file_path) as file:
            info_from_ok_file = json.load(file)

        with self.subTest("test_invalid_file"):
            # Event
            with mock.patch(
                    "asammdf.gui.widgets.functions_manager.QtWidgets.QFileDialog.getOpenFileName"
            ) as mo_getOpenFileName:
                mo_getOpenFileName.return_value = not_ok_file_path, ""
                QTest.mouseClick(self.widget.import_btn, Qt.MouseButton.LeftButton)
                self.processEvents()

            # Evaluate
            mo_getOpenFileName.assert_called_once()
            self.assertEqual(0, self.widget.functions_list.count())

        with self.subTest("test_valid_file"):
            # Event
            with mock.patch(
                    "asammdf.gui.widgets.functions_manager.QtWidgets.QFileDialog.getOpenFileName"
            ) as mo_getOpenFileName:
                mo_getOpenFileName.return_value = ok_file_path, ""
                QTest.mouseClick(self.widget.import_btn, Qt.MouseButton.LeftButton)
                self.processEvents()

            # Evaluate
            mo_getOpenFileName.assert_called()
            self.assertEqual(len(info_from_ok_file), self.widget.functions_list.count())
            for row in range(self.widget.functions_list.count()):
                self.assertIn(self.widget.functions_list.item(row).text(), info_from_ok_file.keys())

    def test_export_btn(self):
        """
        Test scope:
            Check if by pressing 'Save definitions' button,
            all items from functions' list will be saved to a new file.

        Events:
            - Add some functions to the list
            - Press 'Save definitions' button

        Evaluate:
            - Evaluate that all items from functions' list are saved to the file
        """
        # Setup
        # Add some functions
        items_count = randint(1, 10)
        for _ in range(items_count):
            self.widget.add_definition()
        # File with functions definitions
        file_path = path.join(self.test_workspace, "test.def")
        # Event
        with mock.patch(
                "asammdf.gui.widgets.functions_manager.QtWidgets.QFileDialog.getSaveFileName"
        ) as mo_getSaveFileName:
            mo_getSaveFileName.return_value = file_path, ""
            QTest.mouseClick(self.widget.export_btn, Qt.MouseButton.LeftButton)
            self.processEvents()

        with open(file_path) as file:
            info_from_file = json.load(file)

        # Evaluate
        self.assertEqual(len(info_from_file), items_count)
        for row in range(items_count):
            self.assertIn(self.widget.functions_list.item(row).text(), info_from_file.keys())

        file.close()

    def test_check_syntax_btn_0(self):
        """
        Test scope:
            Check if after inserting random text or invalid function in the function definition area,
            by pressing 'Check syntax' button, error dialog will be triggered

        Events:
            - Add some random text in the function definition area
            - Press 'Check syntax' button
            - Add an invalid function in the function definition area
            - Press 'Check syntax' button

        Evaluate:
            - Random text
                > Evaluate that Error dialog was called
            - Invalid function
                > Evaluate that Error dialog was called
        """
        # ok_function = "def" + self.widget.function_definition.placeholderText().rsplit("def", 1)[1]
        not_ok_function = "def f(a=0, t=0): return t if a > t else None"
        with self.subTest("test_not_a_function"):
            # Setup
            self.widget.function_definition.insertPlainText(self.id())
            self.processEvents()

            # Event
            with mock.patch("asammdf.gui.utils.ErrorDialog") as mo_ErrorDialog:
                QTest.mouseClick(self.widget.check_syntax_btn, Qt.MouseButton.LeftButton)

            # Evaluate
            mo_ErrorDialog.assert_called()

        with self.subTest("test_not_ok_function"):
            # Setup
            self.widget.function_definition.insertPlainText(not_ok_function)
            self.processEvents()

            # Event
            with mock.patch("asammdf.gui.utils.ErrorDialog") as mo_ErrorDialog:
                QTest.mouseClick(self.widget.check_syntax_btn, Qt.MouseButton.LeftButton)

            # Evaluate
            mo_ErrorDialog.assert_called()

    def test_check_syntax_btn_1(self):
        """
        Test scope:
            Check if after inserting a valid function in the function definition area,
            by pressing 'Check syntax' button, information dialog will be triggered

        Events:
            - Add a valid function in the function definition area
            - Press 'Check syntax' button

        Evaluate:
            - Evaluate that information dialog was called
        """
        # Setup
        ok_function = "def " + self.widget.function_definition.placeholderText().rsplit("def ", 1)[1]
        self.widget.function_definition.insertPlainText(ok_function)
        self.processEvents()

        # Event
        with mock.patch("asammdf.gui.utils.MessageBox") as mo_MessageBox:
            QTest.mouseClick(self.widget.check_syntax_btn, Qt.MouseButton.LeftButton)

        # Evaluate
        mo_MessageBox.information.assert_called()

    def test_store_btn(self):
        """
        Test scope:
            Check if after inserting a valid function in the function definition area,
            by pressing 'Store function changes' button,
            inserted text is stored in function definition, and two functions cannot have the same name

        Events:
            - Add a valid function in the function definition area
            - Press 'Store function changes' button
            - Add a new item to the functions' list and insert the same function in the function definition area
            - Press 'Store function changes' button
            - Add second item to the functions' list and insert the same function in the function definition area
            - Press 'Store function changes' button


        Evaluate:
            - Without items in a list
                > Evaluate that error dialog was called
            - With an item in the list
                > Evaluate that name of item is identical with the name of function
            - With second item and the same function for both items
                > Evaluate that information messagebox was called
        """
        # Setup
        ok_function = "def " + self.widget.function_definition.placeholderText().rsplit("def ", 1)[1]
        self.widget.function_definition.insertPlainText(ok_function)
        self.processEvents()
        with mock.patch("asammdf.gui.widgets.functions_manager.MessageBox.information") as mo_information:
            with self.subTest("test_without_function"):
                # Event
                QTest.mouseClick(self.widget.store_btn, Qt.MouseButton.LeftButton)
                self.processEvents()

                # Evaluate
                self.mc_ErrorDialog.assert_called()
                mo_information.assert_not_called()

            with self.subTest("test_with_function"):
                # Setup
                self.widget.add_definition()
                self.widget.function_definition.setPlainText(ok_function)

                # Event
                QTest.mouseClick(self.widget.store_btn, Qt.MouseButton.LeftButton)
                self.processEvents()

                # Evaluate
                mo_information.assert_not_called()
                for name, function in self.widget.definitions.items():
                    self.assertEqual(name, self.widget.functions_list.currentItem().text())
                    self.assertEqual(function["definition"], ok_function)

            with self.subTest("test_with_two_identical_named_function"):
                # Setup
                self.widget.add_definition()
                self.widget.function_definition.setPlainText(ok_function)
                # Add new function
                QTest.mouseClick(self.widget.store_btn, Qt.MouseButton.LeftButton)
                self.widget.add_definition()
                self.widget.function_definition.setPlainText(ok_function)

                # Event
                # Try to add the same function second time
                QTest.mouseClick(self.widget.store_btn, Qt.MouseButton.LeftButton)
                self.processEvents()
                # Evaluate
                mo_information.assert_called()
                self.assertIn("Invalid function name", mo_information.call_args[0])

    def test_delete_shortcut(self):
        """
        Test scope:
            Check if by pressing 'Delete' key, selected item will be removed

        Events:
            - Add some functions in the list
            - Clisk on first item
            - Press key 'Delete'

        Evaluate:
            - Evaluate that there is one less item in the functions' list
            - Evaluate that name of the first item was removed from the functions' list

        """
        # Setup
        items_count = randint(2, 10)
        for _ in range(items_count):
            self.widget.add_definition()
        self.processEvents()
        # Quick evaluation
        self.assertEqual(self.widget.functions_list.count(), items_count)
        current_item_text = self.widget.functions_list.currentItem().text()

        # Event
        self.mouseClick_WidgetItem(self.widget.functions_list.item(0))
        QTest.keyClick(self.widget.functions_list, Qt.Key.Key_Delete)
        self.processEvents()

        # Evaluate
        self.assertEqual(items_count - 1, self.widget.functions_list.count())
        self.assertNotIn(current_item_text, [self.widget.functions_list.item(_) for _ in range(items_count - 1)])

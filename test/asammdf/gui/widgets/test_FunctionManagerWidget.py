#!/usr/bin/env python
import json
from os import path
from random import randint
from unittest import mock

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QTreeWidgetItemIterator

from asammdf.gui.widgets.functions_manager import FunctionsManager
from test.asammdf.gui.test_base import TestBase


class TestContextMenu(TestBase):
    """
    ...
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
            Evaluate that one item was added to the functions' list
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
            Evaluate that all items are removed from functions' list
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
        Test scope:# todo
            Check if by pressing 'Delete all' button, all items will be removed from functions' list.

        Events:
            - Add few items in the functions' list
            - Press 'Delete all' button

        Evaluate:
            Evaluate that all items are removed from functions' list
        """
        # Setup
        # File with functions definitions
        file_path = path.join(self.resource, "functions_definitions.def")
        with open(file_path) as file:
            info_from_file = json.load(file)

        # Event
        with mock.patch(
                "asammdf.gui.widgets.functions_manager.QtWidgets.QFileDialog.getOpenFileName"
        ) as mo_getOpenFileName:
            mo_getOpenFileName.return_value = file_path, ""
            QTest.mouseClick(self.widget.import_btn, Qt.MouseButton.LeftButton)
            self.processEvents()

        # Evaluate
        self.assertEqual(len(info_from_file), self.widget.functions_list.count())
        for row in range(self.widget.functions_list.count()):
            self.assertIn(self.widget.functions_list.item(row).text(), info_from_file.keys())

    def test_export_btn(self):
        """
        Test scope: # todo
            Check if by pressing 'Delete all' button, all items will be removed from functions' list.

        Events:
            - Add few items in the functions' list
            - Press 'Delete all' button

        Evaluate:
            Evaluate that all items are removed from functions' list
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

from PySide6 import QtCore, QtTest

from asammdf.gui.dialogs.window_selection_dialog import WindowSelectionDialog
from test.asammdf.gui.test_base import TestBase


class TestWindowSelectionDialog(TestBase):
    def test_RadioButtons(self):
        """
        Test Scope:
            - Ensure that correct radio button is identified
        Events:
            - Check 1st Radio Button
            - Press Apply
            - Check 2nd Radio Button
            - Press Cancel
            - Check 3rd Radio Button
        Evaluate:
            - Evaluate that selected type correspond to RadioButton Checked
        """
        options = ("Plot", "Numeric", "Tabular")
        self.ws = WindowSelectionDialog(options=options)

        for index, option in enumerate(options):
            with self.subTest(f"{self.id()}_{option}"):
                self.ws.showNormal()

                rb = self.ws.selection_layout.itemAt(index).widget()
                self.mouseClick_RadioButton(rb)

                ok, cancel = self.ws.buttonBox.buttons()
                QtTest.QTest.mouseClick(ok, QtCore.Qt.MouseButton.LeftButton)

                self.assertEqual(option, self.ws.selected_type())

    def test_CheckBoxButton(self):
        """
        Purpose:
            - Ensure that state of checkbox is stored.
        Event:
            - Mark CheckBox Button as checked.
            - Mark CheckBox Button as unchecked.
        Evaluate:
            - Evaluate checkbox state
        """
        self.ws = WindowSelectionDialog()

        with self.subTest(f"{self.id()}_Checked"):
            self.ws.showNormal()

            self.mouseClick_CheckboxButton(self.ws.disable_channels)
            ok, cancel = self.ws.buttonBox.buttons()
            QtTest.QTest.mouseClick(ok, QtCore.Qt.MouseButton.LeftButton)

            self.assertTrue(self.ws.disable_new_channels())

        with self.subTest(f"{self.id()}_UnChecked"):
            self.ws.showNormal()

            self.mouseClick_CheckboxButton(self.ws.disable_channels)
            ok, cancel = self.ws.buttonBox.buttons()
            QtTest.QTest.mouseClick(ok, QtCore.Qt.MouseButton.LeftButton)

            self.assertFalse(self.ws.disable_new_channels())

from PySide6 import QtWidgets

from ..ui.windows_selection_dialog import Ui_WindowSelectionDialog


class WindowSelectionDialog(Ui_WindowSelectionDialog, QtWidgets.QDialog):
    def __init__(self, options=("Plot", "Numeric", "Tabular"), default=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        for i, name in enumerate(options):
            radio = QtWidgets.QRadioButton(name)
            self.selection_layout.addWidget(radio)
            if default is None:
                if i == 0:
                    radio.setChecked(True)
            else:
                if name == default:
                    radio.setChecked(True)

    def selected_type(self):
        for i in range(self.selection_layout.count()):
            radio = self.selection_layout.itemAt(i).widget()
            if radio.isChecked():
                return radio.text()

        return ""

    def disable_new_channels(self):
        return self.disable_channels.isChecked()

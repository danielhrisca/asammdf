from PySide6 import QtWidgets

from ..ui.gps_dialog import Ui_GPSDialog
from .advanced_search import AdvancedSearch


class GPSDialog(Ui_GPSDialog, QtWidgets.QDialog):
    def __init__(
        self,
        mdf,
        latitude="",
        longitude="",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self.mdf = mdf

        self.latitude.setText(latitude)
        self.longitude.setText(longitude)

        self.apply_btn.clicked.connect(self._apply)
        self.cancel_btn.clicked.connect(self._cancel)

        self.search_latitude_btn.clicked.connect(self.search_latitude)
        self.search_longitude_btn.clicked.connect(self.search_longitude)

        self.valid = False

    def _apply(self, event):
        if self.latitude.text().strip() and self.longitude.text().strip():
            self.valid = True

        self.close()

    def _cancel(self, event):
        self.close()

    def search_latitude(self, *args):
        dlg = AdvancedSearch(
            self.mdf,
            show_add_window=False,
            show_apply=True,
            show_pattern=False,
            window_title="Search for the latitude channel",
            return_names=True,
            parent=self,
        )
        dlg.setModal(True)
        dlg.exec_()
        result, pattern_window = dlg.result, dlg.pattern_window

        if result:
            self.latitude.setText(list(result)[0])

    def search_longitude(self, *args):
        dlg = AdvancedSearch(
            self.mdf,
            show_add_window=False,
            show_apply=True,
            show_pattern=False,
            window_title="Search for the longitude channel",
            return_names=True,
            parent=self,
        )
        dlg.setModal(True)
        dlg.exec_()
        result, pattern_window = dlg.result, dlg.pattern_window

        if result:
            self.longitude.setText(list(result)[0])

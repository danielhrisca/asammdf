# -*- coding: utf-8 -*-

from PyQt5 import QtWidgets
from natsort import natsorted

from ..ui import resource_rc as resource_rc
from ..ui.tabular_filter import Ui_TabularFilter


class TabularFilter(Ui_TabularFilter, QtWidgets.QWidget):

    def __init__(self, signals, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self.relation.addItems(['AND', 'OR'])
        self.column.addItems(signals)
        self.op.addItems(['>', '>=', '<', '<=', '==', '!='])

        print('ok')
